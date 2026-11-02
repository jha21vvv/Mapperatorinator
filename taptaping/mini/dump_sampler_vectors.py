"""C# 토큰 고르기(MapperGenerator.Process) 검증용: 실제 모델 로짓 + 원본(HF) 처리기를 거친 확률을 뽑는다.

  PYTHONPATH=. python taptaping/mini/dump_sampler_vectors.py --trace trace.json --audio song.mp3 --models exports --out vectors/sampler.txt
  줄 형식(케이스마다 4줄): "lookback_end" / 시퀀스 토큰들 / 로짓(4097개) / 기대 확률(0 이 아닌 것만 '토큰:확률')
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import TopPLogitsWarper

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mapper_mini as mm  # noqa: E402
from osuT5.osuT5.inference.server import build_logits_processors  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tr = json.loads(Path(a.trace).read_text())
tables = json.loads((Path(a.models) / "mapper_tables.json").read_text())
import inference as inf  # noqa: E402
from bench_speedups import make_args  # noqa: E402
args = make_args([f"audio_path={a.audio}", "output_path=.", "gamemode=3", "keycount=4", "device=cpu", "difficulty=2.5"])
_, tok = inf.load_model_with_server(args.model_path, args.train, "cpu", gamemode=3, auto_select_gamemode_model=False,
                                    precision="fp32", attn_implementation="sdpa")
samples = mm.load_audio(a.audio)
k = mm.Tok(tables)
cases = []
for stage, window_cfg, idxs in (("timing", tr["settings"]["timing_window"], [0, 3, 50]), ("map", tr["settings"]["map_window"], [0, 7, 15])):
    lb, la = (float(x) for x in window_cfg.split(","))
    model = mm.OnnxModel(Path(a.models) / stage)
    wins, _, _ = mm.segment(samples, lb, la)
    ws = [w for w in tr["windows"] if w["stage"] == stage]
    for wi in idxs:
        w = ws[wi]
        cross = model.encode(mm.mel_spectrogram(wins[wi]))
        logits, feeds = model.start(cross, w["prompt"])
        seq = list(w["prompt"])
        lb_ms = lb * mm.MS_PER_SEQ if wi != 0 else 0
        lb_end = k.encode("t", int(lb_ms / 10)) if lb_ms > 0 else -1
        procs = build_logits_processors(tok, 1.0, 0, False, 0.9, 0.1, 0.8, 0.8, lb_ms, "cpu")
        procs.append(TopPLogitsWarper(0.9))
        for step in range(min(6, len(w["generated"]))):
            x = torch.tensor(logits, dtype=torch.float32)[None]
            ref = torch.softmax(procs(torch.tensor([seq]), x.clone())[0], -1).numpy()
            cases.append((lb_end, list(seq), logits.astype(np.float32), ref))
            nxt = w["generated"][step]
            seq.append(nxt)
            logits, feeds = model.step(feeds, nxt)
with open(a.out, "w") as f:
    for lb_end, seq, logits, ref in cases:
        f.write(f"{lb_end}\n")
        f.write(" ".join(map(str, seq)) + "\n")
        f.write(" ".join(f"{v:.9g}" for v in logits) + "\n")
        f.write(" ".join(f"{i}:{p:.9g}" for i, p in enumerate(ref) if p > 0) + "\n")
print(f"케이스 {len(cases)}개 → {a.out}")
