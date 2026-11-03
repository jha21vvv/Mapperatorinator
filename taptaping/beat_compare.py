"""폰에 넣을 수 있는 고전 박자 분석 vs AI(Mapperatorinator) 박자선 비교.

박 정렬 작은 모델(GOCT 방식)을 쓰려면 박 격자가 먼저 정확해야 한다. 여기서는
  - 각 방법의 박이 AI 박자선의 박과 얼마나 맞는지 (30ms / 15ms 안 비율, 밀림, 템포 배수 오류)
  - AI 가 만든 노트들이 그 방법의 격자(1/4·1/3 박) 위에 얼마나 올라가는지 (= 그 격자로 AI 수준 악보를 표현할 수 있는지)
를 잰다. AI 박자선도 정답은 아니므로, AI 끼리(시드 다른 실행) 차이를 '바닥 잡음'으로 같이 보여 준다.

  python taptaping/beat_compare.py --audio song.mp3 --ai-timing a.json b.json --ai-notes n1.osu n2.osu
  (a.json: mapper_mini 결과 JSON 의 "timing" 또는 trace 의 "timing_points")
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osu_to_taptaping import parse_osu  # noqa: E402

SR = 22050


def grid_from_timing(tps, end_ms, div=1):
    """박자선 [(offset ms, ms_per_beat, meter)] → 1/div 박 격자 (초)"""
    tps = sorted(tps)
    out = []
    for i, (off, mpb, _m) in enumerate(tps):
        stop = tps[i + 1][0] if i + 1 < len(tps) else end_ms
        t = off
        k = 0
        while t < stop - 1e-6:
            out.append(t / 1000)
            k += 1
            t = off + k * mpb / div
    return np.array(out)


def subdivide(beats, divs=(4, 3)):
    g = set()
    for a, b in zip(beats, beats[1:]):
        for d in divs:
            for k in range(d):
                g.add(round(a + (b - a) * k / d, 5))
    return np.array(sorted(g))


def nearest_dist(sorted_arr, xs):
    idx = np.searchsorted(sorted_arr, xs)
    lo = np.clip(idx - 1, 0, len(sorted_arr) - 1)
    hi = np.clip(idx, 0, len(sorted_arr) - 1)
    return np.minimum(np.abs(xs - sorted_arr[lo]), np.abs(xs - sorted_arr[hi]))


def signed_nearest(sorted_arr, xs):
    idx = np.searchsorted(sorted_arr, xs)
    lo = np.clip(idx - 1, 0, len(sorted_arr) - 1)
    hi = np.clip(idx, 0, len(sorted_arr) - 1)
    a, b = xs - sorted_arr[lo], xs - sorted_arr[hi]
    return np.where(np.abs(a) < np.abs(b), a, b)


def compare(beats, ref_beats, notes_list, lo, hi):
    """beats: 방법의 박 (초). ref_beats: AI 박. lo~hi 초 구간만 (곡 앞뒤 무박 구간 제외)"""
    r = ref_beats[(ref_beats >= lo) & (ref_beats <= hi)]
    b = beats[(beats >= lo - 1) & (beats <= hi + 1)]
    if len(b) < 4:
        return None
    d = nearest_dist(b, r)
    s = signed_nearest(b, r)
    ibi_m, ibi_r = np.median(np.diff(b)), np.median(np.diff(r))
    g = subdivide(b)
    cov = []
    for notes in notes_list:
        n = notes[(notes >= lo) & (notes <= hi)]
        cov.append(float(np.mean(nearest_dist(g, n) <= 0.015)) * 100)
    return {
        "beats": int(len(b)),
        "tempo_ratio": round(float(ibi_r / ibi_m), 3),
        "hit30%": round(float(np.mean(d <= 0.030)) * 100, 1),
        "hit15%": round(float(np.mean(d <= 0.015)) * 100, 1),
        "offset_ms": round(float(np.median(s)) * 1000, 1),
        "notes_on_grid%": round(float(np.mean(cov)), 1),
    }


def method_beats(y, sr, name):
    hop = 128 if "h128" in name else 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if name.startswith("beat_track"):
        tight = 400 if "tight" in name else 100
        _, fr = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, tightness=tight, trim=False)
        beats = librosa.frames_to_time(fr, sr=sr, hop_length=hop)
    elif name.startswith("plp"):
        pulse = librosa.beat.plp(onset_envelope=oenv, sr=sr, hop_length=hop, tempo_min=60, tempo_max=200)
        fr = np.flatnonzero(librosa.util.localmax(pulse))
        beats = librosa.frames_to_time(fr, sr=sr, hop_length=hop)
    elif name.startswith("const"):
        # 고정 BPM: beat_track 박에 직선(박 번호 → 시간) 맞추기 → 등간격 격자
        _, fr = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, tightness=400, trim=False)
        bt = librosa.frames_to_time(fr, sr=sr, hop_length=hop)
        k = np.arange(len(bt))
        a, c = np.polyfit(k, bt, 1)
        beats = c + a * np.arange(-int(c / a) - 1, int((len(y) / sr - c) / a) + 2)
    else:
        raise ValueError(name)
    if "backtrack" in name:
        # 각 박을 근처(±70ms) 온셋 시작점으로 당김 — 온셋 세기 정점이 실제 타격보다 늦게 잡히는 것 보정
        on = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop, backtrack=True, units="time")
        if len(on):
            dd = signed_nearest(np.sort(on), beats)
            beats = np.where(np.abs(dd) <= 0.07, beats - dd, beats)
    if "shift" in name:
        beats = beats - float(name.split("shift")[1].split("_")[0]) / 1000
    return np.sort(beats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ai-timing", nargs="+", required=True, help="JSON (timing 또는 timing_points 키)")
    ap.add_argument("--ai-notes", nargs="*", default=[], help="AI 결과 .osu (노트 격자 적합도용)")
    ap.add_argument("--lo", type=float, default=21, help="비교 시작 초 (앞 무박 구간 제외)")
    ap.add_argument("--hi", type=float, default=193)
    a = ap.parse_args()

    y, sr = librosa.load(a.audio, sr=SR, mono=True)
    end_ms = len(y) / sr * 1000
    ai = []
    for p in a.ai_timing:
        d = json.loads(Path(p).read_text())
        ai.append(d.get("timing") or d.get("timing_points"))
    ai_beats = [grid_from_timing(t, end_ms) for t in ai]
    notes = []
    for p in a.ai_notes:
        c = parse_osu(Path(p).read_text(encoding="utf-8-sig"), p)
        notes.append(np.array(sorted(n["time"] for n in c.notes)))
    ref = ai_beats[0]

    rows = []
    for i, b in enumerate(ai_beats[1:], 1):
        rows.append((f"AI 박자선 (다른 실행 {i})", compare(b, ref, notes, a.lo, a.hi), 0))
    rows.append(("AI 박자선 (기준 자신)", compare(ref, ref, notes, a.lo, a.hi), 0))
    for name in ["beat_track", "beat_track_h128_tight", "beat_track_h128_tight_backtrack", "beat_track_h128_tight_shift50",
                 "plp", "plp_h128", "plp_h128_backtrack", "const_h128", "const_h128_shift50"]:
        t = time.perf_counter()
        b = method_beats(y, sr, name)
        rows.append((name, compare(b, ref, notes, a.lo, a.hi), time.perf_counter() - t))

    keys = ["beats", "tempo_ratio", "hit30%", "hit15%", "offset_ms", "notes_on_grid%"]
    print(f"| 방법 | {' | '.join(keys)} | 초 |")
    print("|---|" + "---|" * (len(keys) + 1))
    for name, m, sec in rows:
        print(f"| {name} | " + (" | ".join(str(m[k]) for k in keys) if m else "실패") + f" | {sec:.2f} |")


if __name__ == "__main__":
    main()
