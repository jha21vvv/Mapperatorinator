"""C# 이식 검증용 자료를 단순한 파일로 뽑는다 (Unity 없이 mono 로 돌리는 MapperReplayTest 가 읽음).

  samples.f32       : 16kHz 모노 정규화 오디오 (float32 리틀엔디언)
  mel0.f32          : 첫 노트 창의 멜 [128×2048] (Python 참조 구현)
  windows.txt       : 창마다 "stage frame_time" 줄 + 프롬프트 줄 + 생성 토큰 줄
  expected_notes.txt: 원본 결과 노트 "시작ms 레인 끝ms(-1=단타)"
  settings.txt      : difficulty, 창 설정

  python taptaping/mini/dump_csharp_vectors.py --trace trace.json --audio song.mp3 --out vectors/
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

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tr = json.loads(Path(a.trace).read_text())
samples = mm.load_audio(a.audio)
samples.astype("<f4").tofile(out / "samples.f32")
mw = tuple(float(x) for x in tr["settings"]["map_window"].split(","))
wins, _, _ = mm.segment(samples, *mw)
mm.mel_spectrogram(wins[0]).astype("<f4").tofile(out / "mel0.f32")
with open(out / "windows.txt", "w") as f:
    for w in tr["windows"]:
        f.write(f"{w['stage']} {w['frame_time']}\n")
        f.write(" ".join(map(str, w["prompt"])) + "\n")
        f.write(" ".join(map(str, w["generated"])) + "\n")
ref = parse_osu(Path(tr["osu_path"]).read_text(encoding="utf-8-sig"), tr["osu_path"]).notes
with open(out / "expected_notes.txt", "w") as f:
    for n in ref:
        st = round(n["time"] * 1000)
        end = round((n["time"] + n["duration"]) * 1000) if n["type"] == 1 else -1
        f.write(f"{st} {n['lane']} {end}\n")
s = tr["settings"]
(out / "settings.txt").write_text(f"{s['difficulty']} {s['timing_window']} {s['map_window']}\n")
(out / "timing_points.txt").write_text("\n".join(f"{o!r} {m!r} {me}" for o, m, me in tr["timing_points"]) + "\n")
print(f"→ {out} (창 {len(tr['windows'])}개, 노트 {len(ref)}개, 샘플 {len(samples)})")
