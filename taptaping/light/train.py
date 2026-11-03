"""가벼운 채보 모델 학습 (CPU 용 — 끊겨도 이어서, 곡을 더해도 이어서).

  python taptaping/light/train.py --data /data/light --ckpt runs/light.pt --hours 36
  (같은 명령을 다시 실행하면 체크포인트에서 이어 학습. 곡을 더 추출한 뒤 다시 실행해도 이어서)

  --threads   : CPU 스레드 수 (기본 = 코어 수)
  --minutes-per-save : 이 간격마다 체크포인트 저장 (기본 20분)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Batcher, load_songs  # noqa: E402
from model import LightChartModel, VOCAB  # noqa: E402


def evaluate(model, batcher, n_batches=20):
    model.eval()
    tot, cnt, correct = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            mel, tok, mask, star = batcher.batch()
            logits = model(mel, tok[:, :-1], star)
            tgt = tok[:, 1:]
            m = mask[:, :-1]
            loss = F.cross_entropy(logits[m], tgt[m], reduction="sum")
            tot += float(loss)
            cnt += int(m.sum())
            correct += int((logits[m].argmax(-1) == tgt[m]).sum())
    model.train()
    return tot / max(cnt, 1), correct / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--minutes-per-save", type=float, default=20)
    ap.add_argument("--limit-songs", type=int, default=0, help="시험용: 곡 수 제한")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    data = Path(a.data)
    t0 = time.perf_counter()
    train = load_songs(data, "train", a.limit_songs or None)
    evals = load_songs(data, "eval", 40)
    print(f"곡: 학습 {len(train)}개 (악보 {sum(len(s.charts) for s in train)}) / 평가 {len(evals)}개 — 불러오기 {time.perf_counter() - t0:.0f}초", flush=True)

    ck = Path(a.ckpt)
    model = LightChartModel(d=a.d, enc_layers=a.layers, dec_layers=a.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    state = {"step": 0, "seen": 0, "history": []}
    if ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        state = c["state"]
        print(f"이어서 학습: {state['step']} 단계부터", flush=True)
    else:
        # 멜 정규화 값 (학습 곡 일부로)
        sample = np.concatenate([s.mel[:: 50].astype(np.float32) for s in train[:300]])
        model.mel_mean.copy_(torch.from_numpy(sample.mean(0)))
        model.mel_std.copy_(torch.from_numpy(sample.std(0) + 1e-3))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 {n_params / 1e6:.2f}M (fp16 약 {n_params * 2 / 1e6:.1f}MB)", flush=True)

    tb = Batcher(train, a.batch, seed=state["step"])
    eb = Batcher(evals, a.batch, seed=1234) if evals else None
    end_time = time.time() + a.hours * 3600
    last_save = time.time()
    t_log, losses = time.perf_counter(), []
    model.train()
    while time.time() < end_time:
        step = state["step"]
        lr = a.lr * min(1.0, (step + 1) / 1000) * (0.5 + 0.5 * math.cos(math.pi * min(1.0, step / 200000)))
        for g in opt.param_groups:
            g["lr"] = max(lr, a.lr * 0.05)
        mel, tok, mask, star = tb.batch()
        logits = model(mel, tok[:, :-1], star)
        m = mask[:, :-1]
        loss = F.cross_entropy(logits[m], tok[:, 1:][m], label_smoothing=0.02)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        state["step"] += 1
        state["seen"] += a.batch
        if state["step"] % 50 == 0:
            dt = time.perf_counter() - t_log
            print(f"단계 {state['step']}  손실 {np.mean(losses):.3f}  {50 * a.batch / dt:.1f} 샘플/초  lr {lr:.2e}", flush=True)
            t_log, losses = time.perf_counter(), []
        if time.time() - last_save > a.minutes_per_save * 60 or time.time() >= end_time:
            ev = evaluate(model, eb) if eb else (float("nan"), float("nan"))
            state["history"].append({"step": state["step"], "eval_loss": ev[0], "eval_acc": ev[1], "time": time.time()})
            print(f"== 평가: 손실 {ev[0]:.3f}, 다음 토큰 정확도 {ev[1] * 100:.1f}%", flush=True)
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                        "config": {"d": a.d, "layers": a.layers}}, tmp)
            os.replace(tmp, ck)
            last_save = time.time()


if __name__ == "__main__":
    main()
