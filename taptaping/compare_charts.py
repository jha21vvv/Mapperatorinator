"""여러 생성 결과(.osu 4K)의 품질 지표를 나란히 비교한다 (표준 라이브러리만).

  python taptaping/compare_charts.py 기준=dirA 개선=dirB ...
  각 dir 아래에서 난이도별 .osu 를 찾는다: dir/d1.5/*.osu 또는 dir_d1.5/*.osu 형태 모두 지원 (--layout 참고)

지표
  노트·롱노트 수, 레인별 비율
  박자 정확도: 노트가 박자 격자(1/4·1/3 박)에서 10ms 이내인 비율 — 박자선은 그 악보의 빨간 선 기준
  동시치기: 같은 시각(5ms 이내) 노트 3개 이상 묶음 수 — 폰 두 손가락 플레이에서 어려운 패턴
  겹침: 같은 레인에서 40ms 이내 연속 노트 수 (창 경계에서 중복 생성되면 늘어남)
  최대 공백: 노트 사이 가장 긴 빈 구간(초) (창 경계에서 노트가 빠지면 늘어남)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osu_to_taptaping import parse_osu  # noqa: E402


def grid_hit_ratio(notes, red_lines, tol_ms=10.0):
    if not red_lines or not notes:
        return 0.0
    lines = sorted(red_lines)
    hit = 0
    for n in notes:
        t = n["time"] * 1000
        start, beat = lines[0]
        for lt, bl in lines:
            if lt <= t + tol_ms:
                start, beat = lt, bl
        rel = (t - start) / beat
        ok = False
        for div in (4, 3):
            x = rel * div
            if abs(x - round(x)) * beat / div <= tol_ms:
                ok = True
                break
        hit += ok
    return hit / len(notes)


def metrics(path: str) -> dict:
    c = parse_osu(Path(path).read_text(encoding="utf-8-sig", errors="replace"), path)
    notes = c.notes
    n = len(notes)
    lanes = [sum(1 for x in notes if x["lane"] == k) for k in range(1, 5)]
    # 동시치기 묶음
    chords3 = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and notes[j + 1]["time"] - notes[i]["time"] <= 0.005:
            j += 1
        if j - i + 1 >= 3:
            chords3 += 1
        i = j + 1
    # 같은 레인 40ms 이내
    last = {}
    dup = 0
    for x in notes:
        p = last.get(x["lane"])
        if p is not None and x["time"] - p <= 0.04:
            dup += 1
        last[x["lane"]] = x["time"]
    times = sorted(x["time"] for x in notes)
    gap = max((b - a for a, b in zip(times, times[1:])), default=0.0)
    return {
        "notes": n,
        "holds": sum(1 for x in notes if x["type"] == 1),
        "lanes%": "/".join(f"{v * 100 // max(n, 1)}" for v in lanes),
        "on_beat%": round(grid_hit_ratio(notes, c.red_lines) * 100, 1),
        "chords3+": chords3,
        "dup40ms": dup,
        "max_gap_s": round(gap, 2),
        "bpm": c.main_bpm(),
        "red_lines": len(c.red_lines),
    }


def find_osu(root: str, diff: str) -> str | None:
    for pat in (f"{root}/d{diff}/*.osu", f"{root}_d{diff}/*.osu", f"{root}/full_d{diff}/*.osu"):
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="이름=경로")
    ap.add_argument("--difficulties", default="1.5,2.5,3.5,4.5")
    a = ap.parse_args()
    runs = [r.split("=", 1) for r in a.runs]
    keys = ["notes", "holds", "lanes%", "on_beat%", "chords3+", "dup40ms", "max_gap_s", "bpm", "red_lines"]
    for d in a.difficulties.split(","):
        print(f"\n## 난이도 {d}")
        print("| 지표 | " + " | ".join(name for name, _ in runs) + " |")
        print("|---|" + "---|" * len(runs))
        cols = []
        for name, root in runs:
            p = find_osu(root, d)
            cols.append(metrics(p) if p else None)
        for k in keys:
            print(f"| {k} | " + " | ".join(str(c[k]) if c else "-" for c in cols) + " |")


if __name__ == "__main__":
    main()
