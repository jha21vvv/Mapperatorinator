"""beat_this 박 → 채보용 박 격자.

  1) 고정 밀림 보정: beat_this 박은 사람 박자선보다 평균 약 30ms 늦게 나온다 (정답 40곡 측정, 10~90% 범위 15~43ms)
  2) 일정 BPM 구간 맞추기: 박 번호 → 시각을 구간별 직선으로 맞춰 박마다의 흔들림을 없앤다 (사람이 박자선을 찍는 방식).
     직선에서 tol 이상 벗어나는 박이 연달아 나오면 거기서 새 구간을 시작한다.
  3) 템포 옥타브 정리: 박 간격이 너무 빠르거나 느리면 반으로/두 배로 (학습 데이터 BPM 대부분 120~240)
"""
from __future__ import annotations

import numpy as np

DEFAULT_SHIFT_SEC = 0.030


def fit_segments(beats: np.ndarray, tol=0.020, min_len=8, run=3):
    """박들을 일정 간격 구간으로 나눠 맞춘 박 시각을 돌려준다."""
    out = []
    i, n = 0, len(beats)
    while i < n:
        j = min(i + min_len, n)
        if j - i < 3:
            out.extend(beats[i:j])
            break
        while True:
            k = np.arange(i, j)
            a, b = np.polyfit(k - i, beats[i:j], 1)
            if j >= n:
                break
            # 다음 run 개 박이 직선 연장에서 tol 넘게 벗어나면 구간 끝
            ahead = np.arange(j, min(j + run, n))
            pred = b + a * (ahead - i)
            if np.all(np.abs(beats[ahead] - pred) > tol):
                break
            j += 1
        k = np.arange(i, j)
        out.extend(b + a * (k - i))
        i = j
    return np.array(out)


def normalize_octave(beats: np.ndarray, lo_bpm=100.0, hi_bpm=250.0):
    ibi = np.median(np.diff(beats))
    bpm = 60 / ibi
    while bpm < lo_bpm:  # 너무 느리면 사이 박 추가 (두 배)
        beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
        bpm *= 2
    while bpm > hi_bpm:  # 너무 빠르면 한 박씩 건너뜀
        beats = beats[::2]
        bpm /= 2
    return beats


# 정답 40곡 측정 (사람 박자선 대비, 30ms 안 박 비율): 보정 없음 44.1% / 밀림 보정 81.6% / +구간 맞추기 79.2% / +옥타브 78.6%
# → 밀림 보정만 기본으로 쓴다 (구간 맞추기·옥타브 정리는 오히려 나빠짐)
def beat_grid(raw_beats, shift=DEFAULT_SHIFT_SEC, fit=False, octave=False):
    b = np.sort(np.asarray(raw_beats, dtype=np.float64)) - shift
    if fit and len(b) >= 8:
        b = fit_segments(b)
    if octave and len(b) >= 8:
        b = normalize_octave(b)
    return b
