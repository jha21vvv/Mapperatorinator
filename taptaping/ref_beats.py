"""모델과 무관한 기준 박 위치를 오디오에서 직접 뽑는다 (librosa 박 추적) → JSON.

  python taptaping/ref_beats.py song.mp3 -o song.beats.json

compare_charts.py --ref-beats 로 넘기면 '실제 박과의 오차' 지표를 계산한다.
librosa 박 추적 자체의 오차가 ±20ms 안팎이라, 지표는 상대 비교용으로 본다.
"""
from __future__ import annotations

import argparse
import json

import librosa
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--start-bpm", type=float, default=120.0)
    a = ap.parse_args()

    y, sr = librosa.load(a.audio, sr=22050, mono=True)
    hop = 128
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, start_bpm=a.start_bpm,
                                           tightness=400)
    bt = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    json.dump({"audio": a.audio, "tempo": float(np.atleast_1d(tempo)[0]), "beats": [round(float(b), 4) for b in bt]},
              open(a.output, "w"), indent=1)
    print(f"박 {len(bt)}개, {bt[0]:.2f}~{bt[-1]:.2f}초, 전체 BPM {float(np.atleast_1d(tempo)[0]):.2f} → {a.output}")


if __name__ == "__main__":
    main()
