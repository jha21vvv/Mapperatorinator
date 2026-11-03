"""L2: 폰용 박자 분석 후보 평가.

기준 박:
  - 데이터셋 평가용 곡(extract.py 가 audio/ 에 남긴 곡): 사람이 찍은 박자선 = 정답
  - 테스트 곡(--test-songs): 정답이 없으므로 AI(Mapperatorinator 박자 모델, ONNX) 박자선을 기준으로
후보:
  - beat_this small0 (2.1M 파라미터, 8MB) / final0 (20M, 77MB — 참고용)
  - librosa beat_track (+ 온셋 시작점 보정)
지표:
  - tempo_ratio : 기준 박 간격 / 후보 박 간격 (1 이 정답, 2·0.5 는 두 배·절반 템포, 1.33 은 4:3 오류)
  - hit30%      : 기준 박 중 후보 박이 30ms 안에 있는 비율
  - hit30_oct%  : 두 배·절반 템포도 맞은 것으로 칠 때 (두 배 템포 격자는 채보에 그대로 쓸 수 있음)
  - offset_ms   : 기준 대비 밀림 (중앙값)

  python taptaping/light/beat_eval.py --data /data/light --test-songs songs/*.mp3 --ai-models exports/ --out beat_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mini"))


def nearest(sorted_arr, xs):
    idx = np.searchsorted(sorted_arr, xs)
    lo = np.clip(idx - 1, 0, len(sorted_arr) - 1)
    hi = np.clip(idx, 0, len(sorted_arr) - 1)
    a, b = xs - sorted_arr[lo], xs - sorted_arr[hi]
    return np.where(np.abs(a) < np.abs(b), a, b)


def variants(ref):
    """기준 박의 두 배(사이 박 추가)·절반(한 박씩 건너뜀, 두 위상) 격자"""
    mids = (ref[:-1] + ref[1:]) / 2
    return [ref, np.sort(np.concatenate([ref, mids])), ref[::2], ref[1::2]]


def score(beats, ref):
    lo, hi = ref[0] + 0.5, ref[-1] - 0.5
    r = ref[(ref >= lo) & (ref <= hi)]
    b = np.sort(beats[(beats >= lo - 2) & (beats <= hi + 2)])
    if len(b) < 4 or len(r) < 4:
        return None
    d = nearest(b, r)
    oct_best = 0.0
    for v in variants(ref):
        vv = v[(v >= lo) & (v <= hi)]
        if len(vv) >= 4:
            # 후보 박 쪽에서 본 일치 (후보 박이 기준 격자 위에 있는가) — 두 배·절반 템포 허용
            oct_best = max(oct_best, float(np.mean(np.abs(nearest(vv, b[(b >= lo) & (b <= hi)])) <= 0.03)))
    near = d[np.abs(d) <= 0.07]
    off = float(np.median(near)) if len(near) else 0.0
    d_al = nearest(b + off, r)  # 일정한 밀림(고정 보정값 하나로 고칠 수 있는 부분)을 빼고 다시
    return {
        "hit30_aligned%": round(float(np.mean(np.abs(d_al) <= 0.03)) * 100, 1),
        "tempo_ratio": round(float(np.median(np.diff(r)) / np.median(np.diff(b))), 3),
        "hit30%": round(float(np.mean(np.abs(d) <= 0.03)) * 100, 1),
        "hit30_oct%": round(oct_best * 100, 1),
        "offset_ms": round(float(np.median(d[np.abs(d) <= 0.07])) * 1000, 1) if np.any(np.abs(d) <= 0.07) else None,
    }


_bt = {}
_wav_cache = {}


def as_wav(path):
    """opus 등은 ffmpeg 로 wav 로 바꿔서 넘긴다 (soundfile 이 일부 opus 를 못 읽음)"""
    path = Path(path)
    if path.suffix.lower() in (".wav", ".mp3"):
        return path
    if path not in _wav_cache:
        import subprocess, tempfile
        tmp = Path(tempfile.gettempdir()) / f"beat_eval_{path.stem}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "22050", str(tmp)], check=True)
        _wav_cache[path] = tmp
    return _wav_cache[path]


def run_method(name, path):
    path = as_wav(path)
    t = time.perf_counter()
    if name.startswith("beat_this"):
        ck = name.split(":")[1]
        if ck not in _bt:
            from beat_this.inference import File2Beats
            _bt[ck] = File2Beats(checkpoint_path=ck, device="cpu", dbn=False)
        beats, _ = _bt[ck](str(path))
        beats = np.asarray(beats, dtype=np.float64)
    else:
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=128)
        _, fr = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=128, tightness=400, trim=False)
        beats = librosa.frames_to_time(fr, sr=sr, hop_length=128)
        if "backtrack" in name:
            on = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=128, backtrack=True, units="time")
            if len(on):
                dd = nearest(np.sort(on), beats)
                beats = np.where(np.abs(dd) <= 0.07, beats - dd, beats)
    return np.sort(beats), time.perf_counter() - t


def ai_beats(path, models: Path):
    import mapper_mini as mm
    tables = json.loads((models / "mapper_tables.json").read_text())
    m = mm.OnnxModel(models / "timing")
    g = mm.Generator(tables, m, m, mm.Settings(seed=7))
    timing, song_len = g.generate_timing_stage(mm.load_audio(str(path)))
    out = []
    for i, tp in enumerate(timing):
        stop = timing[i + 1].offset if i + 1 < len(timing) else song_len
        t = tp.offset
        while t < stop - 1e-6:
            out.append(t / 1000)
            t += tp.mpb
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="extract.py 결과 폴더 (평가용 곡 사람 박자선)")
    ap.add_argument("--max-data-songs", type=int, default=30)
    ap.add_argument("--test-songs", nargs="*", default=[])
    ap.add_argument("--ai-models", help="timing/, mapper_tables.json 폴더 (테스트 곡 기준 박용)")
    ap.add_argument("--methods", default="beat_this:small0,beat_this:final0,librosa,librosa_backtrack")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    methods = a.methods.split(",")
    items = []
    if a.data:
        d = Path(a.data)
        for au in sorted((d / "audio").glob("*.opus"))[: a.max_data_songs]:
            z = np.load(d / "songs" / (au.stem + ".npz"))
            items.append(("사람", au, z["beats"], json.loads(str(z["meta"]))["title"]))
    for p in a.test_songs:
        t = time.perf_counter()
        ref = ai_beats(p, Path(a.ai_models))
        print(f"AI 박자선: {Path(p).name} ({time.perf_counter() - t:.0f}초)", flush=True)
        items.append(("AI", Path(p), ref, Path(p).stem))

    results = []
    for kind, path, ref, title in items:
        row = {"ref": kind, "song": title[:40], "ref_bpm": round(60 / float(np.median(np.diff(ref))), 1)}
        for m in methods:
            try:
                beats, sec = run_method(m, path)
            except Exception as e:  # 곡 하나 실패해도 계속
                print(f"  실패 {m} {title}: {e}", flush=True)
                row[m] = None
                continue
            row[m] = score(beats, ref)
            if row[m] is not None:
                row[m]["sec"] = round(sec, 1)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))

    for kind in ("사람", "AI"):
        rows = [r for r in results if r["ref"] == kind]
        if not rows:
            continue
        print(f"\n## 기준: {kind} 박자선 ({len(rows)}곡)")
        print("| 방법 | 템포 맞음(±3%) | 두배/절반 | 기타(4:3 등) | hit30% 평균 | 밀림 보정 후 hit30% | hit30_oct% 평균 | 밀림 중앙(ms) | 초/곡 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for m in methods:
            ms = [r[m] for r in rows if r.get(m)]
            if not ms:
                continue
            ok = sum(1 for x in ms if abs(x["tempo_ratio"] - 1) <= 0.03)
            octv = sum(1 for x in ms if abs(x["tempo_ratio"] - 2) <= 0.06 or abs(x["tempo_ratio"] - 0.5) <= 0.015)
            other = len(ms) - ok - octv
            offs = [x["offset_ms"] for x in ms if x["offset_ms"] is not None]
            print(f"| {m} | {ok}/{len(ms)} | {octv} | {other} | {np.mean([x['hit30%'] for x in ms]):.1f} | "
                  f"{np.mean([x['hit30_aligned%'] for x in ms]):.1f} | "
                  f"{np.mean([x['hit30_oct%'] for x in ms]):.1f} | {np.median(offs) if offs else '-'} | "
                  f"{np.mean([x['sec'] for x in ms]):.1f} |")


if __name__ == "__main__":
    main()
