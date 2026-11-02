"""여러 생성 결과(.osu 4K)의 품질 지표를 나란히 비교한다 (표준 라이브러리만).

  python taptaping/compare_charts.py --ref-beats song.beats.json \
      "기준=/path/full_d{d}/*.osu" "①②=/path/b12_s7/d{d}/*.osu,/path/b12_s8/d{d}/*.osu"

  각 실행은 이름=경로패턴. {d} 는 난이도로 바뀐다. 쉼표로 여러 시드를 주면 평균(최소~최대)을 보여준다.

지표
  notes / holds        노트·롱노트 수
  ref_beat_err_ms      실제 박(ref_beats.py)마다, 악보 박자 격자(1/4박 포함)에서 가장 가까운 선까지 거리의 중앙값
  ref_beat_ok%         그 거리가 30ms 이내인 실제 박 비율
  note_on_ref%         노트가 실제 박을 1/4·1/3 로 나눈 격자에서 30ms 이내인 비율 (모델 박자와 무관한 음악 기준)
  offset_ms            실제 박 대비 악보 박자선(정박)의 부호 있는 차이 중앙값 — 악보 전체가 밀렸는지 (+ 면 악보가 빠름)
  odd_bpm              그 구간의 실제 템포와 맞지 않는 박자선 수 (배수·절반·3/2 등 흔한 비율이 아닌 것)
  red_lines            박자선 수 (참고용 — 템포가 바뀌는 곡이면 많아도 정상)
  chords3+             노트 3개 이상 동시치기 묶음 수 (폰 두 엄지 플레이에서 어려움)
  dup40ms              같은 레인 40ms 이내 연속 노트 수
  max_gap_s            가장 긴 빈 구간(초)
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osu_to_taptaping import parse_osu  # noqa: E402

TOL = 0.030
COMMON_RATIOS = (0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, 4 / 3, 1.5, 2.0, 3.0, 4.0)


def grid_lines(red_lines, end_s, div=4):
    """박자선 → 1/div 박 격자 시각 목록 (초)"""
    lines = sorted(red_lines)
    out = []
    for i, (t_ms, bl) in enumerate(lines):
        stop_ms = lines[i + 1][0] if i + 1 < len(lines) else end_s * 1000
        step = bl / div
        t = t_ms
        while t < stop_ms - 1e-6:
            out.append(t / 1000)
            t += step
    return out


def nearest(sorted_vals, x):
    i = bisect.bisect_left(sorted_vals, x)
    best = float("inf")
    for j in (i - 1, i):
        if 0 <= j < len(sorted_vals):
            best = min(best, abs(sorted_vals[j] - x))
    return best


def ref_grid(beats, divs=(4, 3)):
    g = set()
    for a, b in zip(beats, beats[1:]):
        for div in divs:
            for k in range(div):
                g.add(round(a + (b - a) * k / div, 4))
    g.add(beats[-1])
    return sorted(g)


def local_tempo(beats, t):
    i = bisect.bisect_left(beats, t)
    lo, hi = max(0, i - 4), min(len(beats) - 1, i + 4)
    if hi - lo < 2:
        return None
    return 60.0 * (hi - lo) / (beats[hi] - beats[lo])


def metrics(path: str, ref: list[float] | None) -> dict:
    c = parse_osu(Path(path).read_text(encoding="utf-8-sig", errors="replace"), path)
    notes = c.notes
    n = len(notes)
    times = sorted(x["time"] for x in notes)
    chords3, i = 0, 0
    while i < n:
        j = i
        while j + 1 < n and notes[j + 1]["time"] - notes[i]["time"] <= 0.005:
            j += 1
        chords3 += (j - i + 1) >= 3
        i = j + 1
    last, dup = {}, 0
    for x in notes:
        p = last.get(x["lane"])
        dup += p is not None and x["time"] - p <= 0.04
        last[x["lane"]] = x["time"]
    m = {
        "notes": n,
        "holds": sum(1 for x in notes if x["type"] == 1),
        "chords3+": chords3,
        "dup40ms": dup,
        "max_gap_s": round(max((b - a for a, b in zip(times, times[1:])), default=0.0), 2),
        "red_lines": len(c.red_lines),
    }
    if ref:
        end = max(times[-1] if times else 0, ref[-1]) + 1
        grid = grid_lines(c.red_lines, end)
        errs = [nearest(grid, b) for b in ref]
        m["ref_beat_err_ms"] = round(statistics.median(errs) * 1000, 1)
        m["ref_beat_ok%"] = round(100 * sum(e <= TOL for e in errs) / len(errs), 1)
        rg = ref_grid(ref)
        in_range = [t for t in times if ref[0] - TOL <= t <= ref[-1] + TOL]
        m["note_on_ref%"] = round(100 * sum(nearest(rg, t) <= TOL for t in in_range) / max(len(in_range), 1), 1)
        odd = 0
        for t_ms, bl in c.red_lines:
            lt = local_tempo(ref, t_ms / 1000)
            if lt is None:
                continue
            r = (60000 / bl) / lt
            odd += not any(abs(r / cr - 1) <= 0.04 for cr in COMMON_RATIOS)
        m["odd_bpm"] = odd
        beat_grid = grid_lines(c.red_lines, end, div=1)

        def signed(x):
            k = bisect.bisect_left(beat_grid, x)
            cand = [beat_grid[j] for j in (k - 1, k) if 0 <= j < len(beat_grid)]
            return min((x - v for v in cand), key=abs)

        m["offset_ms"] = round(statistics.median(signed(b) for b in ref) * 1000, 1)
    return m


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:g}"
    mean = sum(vals) / len(vals)
    return f"{mean:.1f} ({min(vals):g}~{max(vals):g})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help='이름=경로패턴[,경로패턴...]  ({d} = 난이도)')
    ap.add_argument("--difficulties", default="1.5,2.5,3.5,4.5")
    ap.add_argument("--ref-beats", help="ref_beats.py 결과 JSON")
    ap.add_argument("--json", help="모든 수치를 JSON 으로도 저장")
    ap.add_argument("--ref-shift-ms", type=float, default=0.0,
                    help="기준 박을 이만큼 앞당김 (librosa 박 추적의 고정 지연 보정용)")
    a = ap.parse_args()
    ref = [b - a.ref_shift_ms / 1000 for b in json.load(open(a.ref_beats))["beats"]] if a.ref_beats else None
    runs = [(r.split("=", 1)[0], r.split("=", 1)[1].split(",")) for r in a.runs]
    keys = ["notes", "holds", "note_on_ref%", "ref_beat_err_ms", "ref_beat_ok%", "offset_ms", "odd_bpm", "red_lines",
            "chords3+", "dup40ms", "max_gap_s"]
    if not ref:
        keys = [k for k in keys if k not in ("note_on_ref%", "ref_beat_err_ms", "ref_beat_ok%", "offset_ms", "odd_bpm")]
    dump = {}
    for d in a.difficulties.split(","):
        print(f"\n## 난이도 {d}")
        print("| 지표 | " + " | ".join(name for name, _ in runs) + " |")
        print("|---|" + "---|" * len(runs))
        per_run = []
        for name, pats in runs:
            ms = []
            for pat in pats:
                hits = sorted(glob.glob(pat.replace("{d}", d)))
                if hits:
                    ms.append(metrics(hits[0], ref))
            per_run.append(ms)
            dump.setdefault(name, {})[d] = ms
        for k in keys:
            print(f"| {k} | " + " | ".join(fmt([m.get(k) for m in ms]) for ms in per_run) + " |")
    if a.json:
        json.dump(dump, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
