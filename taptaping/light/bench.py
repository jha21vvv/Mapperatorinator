"""가벼운 엔진 단계별 시간 측정 (폰 예상의 근거).

  python taptaping/light/bench.py --audio song.mp3 --ckpt runs/light.pt --threads 1 4
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
from beat_grid import beat_grid  # noqa: E402
from extract import SR, SUB, beat_aligned_mel, decode  # noqa: E402
from model import BOS, SEP, HOLD0, WIN, LightChartModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--data", help="평균 토큰 수 계산용 (extract 결과)")
    a = ap.parse_args()
    res = {}
    for th in a.threads:
        torch.set_num_threads(th)
        r = {}
        t = time.perf_counter()
        y = decode(Path(a.audio).read_bytes())
        r["decode_s"] = time.perf_counter() - t
        song_sec = len(y) / SR
        from beat_this.inference import File2Beats
        f2b = File2Beats(checkpoint_path="small0", device="cpu", dbn=False)
        t = time.perf_counter()
        raw, _ = f2b(a.audio)
        r["beat_this_s"] = time.perf_counter() - t
        beats = beat_grid(raw)
        t = time.perf_counter()
        centers = (beats[:-1, None] + (beats[1:] - beats[:-1])[:, None] * (np.arange(SUB)[None, :] / SUB)).reshape(-1)
        mel = beat_aligned_mel(y, centers)
        r["mel_s"] = time.perf_counter() - t
        n_blocks = (len(beats) - 1) // 2
        model = LightChartModel()
        if a.ckpt:
            model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"])
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(mel[:WIN].astype(np.float32))[None]
            for _ in range(3):
                mem = model.encode(x)
            t = time.perf_counter()
            for _ in range(20):
                mem = model.encode(x)
            r["encode_ms_per_block"] = (time.perf_counter() - t) / 20 * 1000
            for L in (10, 40, 80):
                seq = torch.tensor([[BOS] + [10] * (L - 3) + [SEP, HOLD0]])
                t = time.perf_counter()
                for _ in range(20):
                    model.decode(seq, mem, torch.tensor([3.0]))
                r[f"decode_ms_len{L}"] = (time.perf_counter() - t) / 20 * 1000
        r["song_sec"] = song_sec
        r["blocks"] = n_blocks
        res[th] = r
        print(th, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    if a.data:
        from data import load_songs, block_tokens
        songs = load_songs(Path(a.data), "train", 150)
        per_block = []
        for s in songs:
            for c in s.charts:
                for k in range(0, s.n_beats, 2):
                    per_block.append((len(block_tokens(c["by_pos"], k * SUB, 2 * SUB)), c["star"]))
        arr = np.array(per_block)
        for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 10)):
            m = (arr[:, 1] >= lo) & (arr[:, 1] < hi)
            print(f"★{lo}~{hi}: 2박당 노트 토큰 평균 {arr[m, 0].mean():.1f} (최대 {arr[m, 0].max():.0f})")


if __name__ == "__main__":
    main()
