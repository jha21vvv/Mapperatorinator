"""taptaping 악보 JSON(ChartExportData)을 그림으로 — 난이도 4개를 나란히, 레인 4개, 시간은 아래→위.

python taptaping/preview_chart.py chart.json -o preview.png --start 0 --end 30
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]  # 한글 표시용 (없으면 기본 글꼴)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

DIFFS = ("easy", "normal", "hard", "insane")
LANE_COLORS = ("#ff6b8b", "#5ec8ff", "#5ec8ff", "#ff6b8b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chart")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=30.0)
    args = ap.parse_args()

    data = json.load(open(args.chart, encoding="utf-8"))
    bpm = data.get("bpm") or 0
    fig, axes = plt.subplots(1, 4, figsize=(10, 12), sharey=True, facecolor="#14161c")
    for ax, d in zip(axes, DIFFS):
        notes = [n for n in data.get(f"{d}Notes", []) if args.start <= n["time"] <= args.end]
        ax.set_facecolor("#1d2029")
        for lane in range(5):
            ax.axvline(lane + 0.5, color="#33384a", lw=1)
        if bpm > 0:  # 박자선 (첫 노트 기준이 아니라 0초 기준 — 대략 보기용)
            beat = 60.0 / bpm
            t = args.start - (args.start % beat)
            while t <= args.end:
                ax.axhline(t, color="#2a2e3b", lw=0.6)
                t += beat
        for n in notes:
            x = n["lane"] - 0.4
            col = LANE_COLORS[n["lane"] - 1]
            if n["type"] == 1 and n["duration"] > 0:
                ax.add_patch(Rectangle((x + 0.15, n["time"]), 0.5, n["duration"], color=col, alpha=0.45, lw=0))
            ax.add_patch(Rectangle((x, n["time"] - 0.04), 0.8, 0.08, color=col, lw=0))
        all_n = data.get(f"{d}Notes", [])
        nps = len(all_n) / max(all_n[-1]["time"] - all_n[0]["time"], 1e-6) if all_n else 0
        ax.set_title(f"{d}\n{len(all_n)}개 · 초당 {nps:.1f}", color="white", fontsize=11)
        ax.set_xlim(0.5, 4.5)
        ax.set_xticks([1, 2, 3, 4])
        ax.tick_params(colors="#9aa0b4")
    axes[0].set_ylim(args.start, args.end)
    axes[0].set_ylabel("시간 (초)", color="#9aa0b4")
    fig.suptitle(f"{data.get('title', '')}  ·  BPM {bpm:g}  ·  {args.start:g}~{args.end:g}초", color="white")
    fig.tight_layout()
    fig.savefig(args.output, dpi=90, facecolor=fig.get_facecolor())
    print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
