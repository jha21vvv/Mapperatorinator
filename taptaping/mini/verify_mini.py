"""mapper_mini.py 를 원본 기록(trace_generation.py 결과)과 대조 검증.

  1. 멜 스펙트로그램: 원본 모델 안의 torchaudio 멜과 비교
  2. 창 나누기: 창 시작 시각이 원본과 같은지
  3. 원본이 생성한 토큰을 그대로 넣어 가며(teacher forcing) 창마다 프롬프트가 토큰 하나까지 같은지
  4. 박자선: 원본 박 이벤트 → generate_timing 결과가 원본 박자선과 같은지
  5. 노트: 원본 결과 .osu 의 노트와 같은지

  PYTHONPATH=. python taptaping/mini/verify_mini.py --trace trace.json --audio song.mp3 --tables mapper_tables.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mapper_mini as mm  # noqa: E402
from osu_to_taptaping import parse_osu  # noqa: E402


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else '틀림'}] {name} {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--skip-mel", action="store_true")
    a = ap.parse_args()
    tr = json.loads(Path(a.trace).read_text())
    tables = json.loads(Path(a.tables).read_text())
    st = tr["settings"]
    tw = tuple(float(x) for x in st["timing_window"].split(","))
    mw = tuple(float(x) for x in st["map_window"].split(","))
    s = mm.Settings(difficulty=st["difficulty"], timing_window=tw, map_window=mw)
    g = mm.Generator(tables, None, None, s)
    all_ok = True

    samples = mm.load_audio(a.audio)
    win_t, times_t, song_length = mm.segment(samples, *tw)
    win_m, times_m, _ = mm.segment(samples, *mw)
    all_ok &= check("곡 길이", abs(song_length - tr["song_length_ms"]) < 1e-6, f"{song_length} vs {tr['song_length_ms']}")
    all_ok &= check("박자 창 시각", times_t == [int(x) for x in tr["timing_frame_times"]], f"{len(times_t)}개")
    all_ok &= check("노트 창 시각", times_m == [int(x) for x in tr["map_frame_times"]], f"{len(times_m)}개")

    if not a.skip_mel:
        import torch
        from osuT5.osuT5.model import Mapperatorinator
        model = Mapperatorinator.from_pretrained("OliBomby/Mapperatorinator-v32-mini", subfolder="gamemode=3")
        errs = []
        for w in (win_m[0], win_m[len(win_m) // 2]):
            ref = model.spectrogram(torch.from_numpy(np.ascontiguousarray(w))[None])[0].T.numpy()  # [128, 2048]
            mine = mm.mel_spectrogram(w)
            errs.append(float(np.abs(ref - mine).max() / max(np.abs(ref).max(), 1e-9)))
        all_ok &= check("멜 스펙트로그램", max(errs) < 1e-4, f"상대 오차 최대 {max(errs):.2e}")

    # ── 박자 단계 재연 ──────────────────────────────────────────────
    tws = [w for w in tr["windows"] if w["stage"] == "timing"]
    mws = [w for w in tr["windows"] if w["stage"] == "map"]
    cls = g.class_tokens(song_length)
    ctx = mm.Context("timing")
    bad = 0
    for i, w in enumerate(tws):
        ft = w["frame_time"]
        prompt = g.build_prompt(cls, [ctx], ft, song_length)
        if prompt != w["prompt"]:
            bad += 1
            if bad <= 2:
                print("   원본", w["prompt"][:40], len(w["prompt"]))
                print("   재연", prompt[:40], len(prompt))
        trim_lb = i != 0 and tw[0] > 0
        trim_la = i != len(tws) - 1
        g.add_tokens(ctx, list(w["generated"]), ft, trim_lb, trim_la, *tw)
    all_ok &= check("박자 프롬프트 (teacher forcing)", bad == 0, f"{len(tws) - bad}/{len(tws)} 창 일치")

    beat_events = [e for e in ctx.events if e[0] in ("t", "beat", "measure", "timing_point")]
    ref_ev = [(e[0], e[1]) for e in tr["timing_events"]]
    all_ok &= check("박 이벤트", beat_events == ref_ev, f"{len(beat_events)} vs {len(ref_ev)}")
    timing = mm.generate_timing(ref_ev, tables["timing_leniency"])
    ref_tp = tr["timing_points"]
    same_tp = len(timing) == len(ref_tp) and all(abs(a.offset - b[0]) < 1e-6 and abs(a.mpb - b[1]) < 1e-6 and a.meter == b[2]
                                                for a, b in zip(timing, ref_tp))
    all_ok &= check("박자선 (generate_timing)", same_tp, f"{len(timing)} vs {len(ref_tp)}")
    if not same_tp:
        for a_, b_ in list(zip(timing, ref_tp))[:6]:
            print("   ", (a_.offset, a_.mpb, a_.meter), b_)

    # ── 노트 단계 재연 ──────────────────────────────────────────────
    tev, ttimes = mm.parse_timing(timing, song_length)
    given = mm.Context("timing", tev, ttimes, True)
    mctx = mm.Context("map")
    bad = 0
    for i, w in enumerate(mws):
        ft = w["frame_time"]
        prompt = g.build_prompt(cls, [given, mctx], ft, song_length)
        if prompt != w["prompt"]:
            bad += 1
            if bad <= 2:
                diff_at = next((k for k in range(min(len(prompt), len(w["prompt"]))) if prompt[k] != w["prompt"][k]), None)
                print(f"   창 {i} 처음 다른 위치 {diff_at}: 원본 {w['prompt'][max(0, (diff_at or 0) - 3):(diff_at or 0) + 5]} "
                      f"재연 {prompt[max(0, (diff_at or 0) - 3):(diff_at or 0) + 5]} 길이 {len(w['prompt'])}/{len(prompt)}")
        trim_lb = i != 0 and mw[0] > 0
        trim_la = i != len(mws) - 1
        g.add_tokens(mctx, list(w["generated"]), ft, trim_lb, trim_la, *mw)
    all_ok &= check("노트 프롬프트 (teacher forcing)", bad == 0, f"{len(mws) - bad}/{len(mws)} 창 일치")

    notes = mm.events_to_notes(mctx.events, timing, 4)
    ref = parse_osu(Path(tr["osu_path"]).read_text(encoding="utf-8-sig"), tr["osu_path"]).notes
    ref_n = [[round(n["time"] * 1000), n["lane"], round((n["time"] + n["duration"]) * 1000) if n["type"] == 1 else None] for n in ref]
    key = lambda n: (n[0], n[1], n[2] or -1)  # noqa: E731
    mine_s, ref_s = sorted(notes, key=key), sorted(ref_n, key=key)
    same = mine_s == ref_s
    all_ok &= check("노트 (스냅 보정·롱노트 포함)", same, f"{len(notes)} vs {len(ref_n)}")
    if not same:
        diffs = [(x, y) for x, y in zip(mine_s, ref_s) if x != y][:5]
        print("   ", diffs)
    print("\n전체:", "통과" if all_ok else "실패 있음")


if __name__ == "__main__":
    main()
