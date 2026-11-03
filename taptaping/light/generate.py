"""가벼운 모델로 악보 생성 + 사람 악보와 비교.

  # 데이터셋 평가 곡 (사람 박자선 사용) — 사람 악보와 F1 비교
  python taptaping/light/generate.py --ckpt runs/light.pt --eval-data /data/light --n 20

  # 새 곡 (박은 beat_this small0 으로 찾음) → .osu / taptaping JSON
  python taptaping/light/generate.py --ckpt runs/light.pt --audio song.mp3 --stars 1.5,2.5,3.5,4.5 --out out_dir
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Song, block_tokens, held_state  # noqa: E402
from model import (ACT0, BLOCK_BEATS, BOS, CTX_BEATS, EOS, HOLD0, MAX_CTX_TOKENS, MAX_OUT_TOKENS, N_ACT, N_TIME, SEP,  # noqa: E402
                   SUB, TIME0, WIN, LightChartModel, action_states)


def load_model(path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    cfg = c.get("config", {})
    m = LightChartModel(d=cfg.get("d", 192), enc_layers=cfg.get("layers", 3), dec_layers=cfg.get("layers", 3))
    m.load_state_dict(c["model"])
    m.eval()
    return m, c.get("state", {})


@torch.no_grad()
def generate_chart(model, mel_all: np.ndarray, n_beats: int, star: float, temperature=0.9, top_p=0.9, seed=0):
    """곡 전체를 2박씩 생성 → notes [(칸 위치, 레인, 종류)]"""
    g = torch.Generator().manual_seed(seed)
    by_pos, holds, open_ = {}, [], {}
    star_t = torch.tensor([star], dtype=torch.float32)
    floor = np.log(1e-6)
    for k in range(0, n_beats, BLOCK_BEATS):
        s0 = (k - CTX_BEATS) * SUB
        mel = np.full((WIN, mel_all.shape[1]), floor, np.float32)
        a, b = max(s0, 0), min(s0 + WIN, mel_all.shape[0])
        if b > a:
            mel[a - s0: b - s0] = mel_all[a:b]
        memory = model.encode(torch.from_numpy(mel)[None])
        gen_pos = k * SUB
        ctx = block_tokens(by_pos, gen_pos - CTX_BEATS * SUB, CTX_BEATS * SUB)[-MAX_CTX_TOKENS:]
        held = held_state([(c, s, e) for c, s, e in holds] + [(c, s, 10 ** 9) for c, s in open_.items()], gen_pos)
        seq = [BOS] + ctx + [SEP, held]
        last_time = -1
        for _ in range(MAX_OUT_TOKENS):
            logits = model.decode(torch.tensor([seq]), memory, star_t)[0, -1].clone()
            expect_time = (len(seq) - len(ctx) - 3) % 2 == 0  # 시간 → 행동 → 시간 … 순서 강제
            allowed = torch.zeros_like(logits, dtype=torch.bool)
            if expect_time:
                allowed[EOS] = True
                allowed[TIME0 + last_time + 1: TIME0 + N_TIME] = True  # 시간은 앞으로만
            else:
                allowed[ACT0: ACT0 + N_ACT] = True
            logits[~allowed] = float("-inf")
            probs = torch.softmax(logits / temperature, -1)
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, 0) - sp < top_p
            probs2 = torch.zeros_like(probs)
            probs2[si[keep]] = sp[keep]
            tok = int(torch.multinomial(probs2 / probs2.sum(), 1, generator=g))
            seq.append(tok)
            if tok == EOS:
                break
            if expect_time:
                last_time = tok - TIME0
            else:
                pos = gen_pos + last_time
                st = action_states(tok)
                cur = by_pos.setdefault(pos, [0, 0, 0, 0])
                for col in range(4):
                    if st[col] == 0:
                        continue
                    if st[col] == 2:
                        if col in open_:
                            continue  # 이미 누르는 중이면 무시
                        open_[col] = pos
                    elif st[col] == 3:
                        if col not in open_:
                            continue  # 짝 없는 롱끝 무시
                        holds.append((col, open_.pop(col), pos))
                    elif col in open_:
                        continue  # 누르는 중인 키에 단타 무시
                    cur[col] = st[col]
    for col, s in open_.items():  # 끝나지 않은 롱노트는 단타로
        if by_pos.get(s):
            by_pos[s][col] = 1
    notes = []
    for pos in sorted(by_pos):
        for col, kind in enumerate(by_pos[pos]):
            if kind:
                notes.append((pos, col, kind))
    return notes


def onsets(notes):
    return {(p, c) for p, c, k in notes if k in (1, 2)}


def f1(pred, ref, tol=1):
    """레인까지 맞힌 노트 F1 (위치 ±tol 칸 = 1/48 박)"""
    ref_l = sorted(ref)
    used, tp = set(), 0
    ref_by = {}
    for p, c in ref_l:
        ref_by.setdefault(c, []).append(p)
    for p, c in sorted(pred):
        for q in ref_by.get(c, []):
            if abs(q - p) <= tol and (q, c) not in used:
                used.add((q, c))
                tp += 1
                break
    prec = tp / max(len(pred), 1)
    rec = tp / max(len(ref), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


def timing_f1(pred, ref, tol=1):
    """레인 무시, 시간 위치만 F1 (GOCT 논문 지표와 비슷). 같은 칸의 동시 노트는 하나로 봄."""
    pp = sorted({p for p, _c in pred})
    rr = sorted({p for p, _c in ref})
    used, tp, j = set(), 0, 0
    for p in pp:
        while j < len(rr) and rr[j] < p - tol:
            j += 1
        for q in rr[j: j + 3]:
            if abs(q - p) <= tol and q not in used:
                used.add(q)
                tp += 1
                break
    prec, rec = tp / max(len(pp), 1), tp / max(len(rr), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-data")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--audio")
    ap.add_argument("--stars", default="2.5")
    ap.add_argument("--out")
    a = ap.parse_args()
    model, st = load_model(a.ckpt)
    print(f"모델: {st.get('step')} 단계 학습", flush=True)

    if a.eval_data:
        d = Path(a.eval_data)
        rows = []
        for p in sorted((d / "songs").glob("*.npz")):
            meta = json.loads(str(np.load(p)["meta"]))
            if not meta.get("eval"):
                continue
            song = Song(p)
            for ch in song.charts:
                t = time.perf_counter()
                pred = generate_chart(model, song.mel.astype(np.float32), song.n_beats, ch["star"])
                ref = [(pos, c, k) for pos, st4 in ch["by_pos"].items() for c, k in enumerate(st4) if k]
                f, pr, rc = f1(onsets(pred), onsets(ref))
                tf = timing_f1(onsets(pred), onsets(ref))
                rows.append({"song": meta["title"][:30], "star": round(ch["star"], 2), "ref_notes": len(onsets(ref)),
                             "pred_notes": len(onsets(pred)), "timing_f1": round(tf * 100, 1), "lane_f1": round(f * 100, 1),
                             "sec": round(time.perf_counter() - t, 1)})
                print(rows[-1], flush=True)
            if len({r["song"] for r in rows}) >= a.n:
                break
        if rows:
            print(f"\n평균: 시간 F1 {np.mean([r['timing_f1'] for r in rows]):.1f}%, 레인까지 F1 {np.mean([r['lane_f1'] for r in rows]):.1f}%, "
                  f"노트 수 비(생성/사람) {np.mean([r['pred_notes'] / max(r['ref_notes'], 1) for r in rows]):.2f}, "
                  f"악보당 {np.mean([r['sec'] for r in rows]):.1f}초")


if __name__ == "__main__":
    main()
