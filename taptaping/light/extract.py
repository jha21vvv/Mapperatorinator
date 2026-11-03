"""가벼운 엔진용 데이터 추출: osu-beatmaps(WebDataset) 조각에서 4K mania 곡만 골라 학습 자료로 만든다.

곡 하나 → <out>/songs/<audio_hash 앞 16자>.npz
  mel      : 박 정렬 로그 멜 [박 수 × 48, 80] float16  (한 박을 48칸으로 — GOCT 방식, 템포와 무관한 고정 칸)
  beats    : 박 시각(초) [박 수 + 1]  (사람이 찍은 박자선에서 계산 — 정답 격자)
  charts   : JSON 문자열 — 악보마다 {star, version, notes:[[칸 위치, 레인 0~3, 종류 1단타/2롱시작/3롱끝], ...]}
  meta     : JSON 문자열 — 제목, 길이, BPM, 박자선, 양자화 오차 통계
평가용 곡(해시 기준 5%)은 박자 분석 평가(L2)를 위해 오디오(opus)도 <out>/audio/ 에 남긴다.

  python taptaping/light/extract.py --shards 37 3 12 ... --out /data/light
  (조각을 하나씩 받아 처리하고 지운다. 이미 처리한 조각은 건너뛴다)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/datasets/project-riz/osu-beatmaps/resolve/main/compressed/data-{:06d}.tar"
SR = 16000
N_FFT = 512
N_MELS = 80
SUB = 48  # 한 박을 나누는 칸 수


# ── 오디오 · 멜 ────────────────────────────────────────────────────────────
def decode(data: bytes) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                       input=data, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_filterbank():
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    pts = 700.0 * (10 ** (np.linspace(_hz_to_mel(20.0), _hz_to_mel(SR / 2), N_MELS + 2) / 2595.0) - 1)
    fb = np.zeros((N_FFT // 2 + 1, N_MELS), np.float32)
    for m in range(N_MELS):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[:, m] = np.maximum(0, np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)))
    return fb


FB = mel_filterbank()
WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)


def beat_aligned_mel(y: np.ndarray, centers_sec: np.ndarray) -> np.ndarray:
    """주어진 시각마다 창(512)을 가운데 두고 멜 → log. centers 는 박을 48등분한 시각들."""
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad + 1))
    idx = np.clip(np.round(centers_sec * SR).astype(np.int64), 0, len(y))
    out = np.empty((len(idx), N_MELS), np.float32)
    for s in range(0, len(idx), 4096):  # 메모리 아끼려고 나눠서
        frames = yp[idx[s:s + 4096, None] + np.arange(N_FFT)[None, :]] * WIN
        p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        out[s:s + 4096] = np.log(p @ FB + 1e-6)
    return out


# ── .osu 읽기 ──────────────────────────────────────────────────────────────
def parse_osu(text: str):
    sec = ""
    reds, notes, keys = [], [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1]
            continue
        if sec == "Difficulty" and line.startswith("CircleSize"):
            keys = float(line.split(":", 1)[1])
        elif sec == "TimingPoints":
            p = line.split(",")
            if len(p) >= 2:
                t, bl = float(p[0]), float(p[1])
                unin = int(p[6]) if len(p) > 6 else (1 if bl > 0 else 0)
                meter = int(p[2]) if len(p) > 2 and p[2].strip() else 4
                if unin == 1 and bl > 0:
                    reds.append((t, bl, meter))
        elif sec == "HitObjects":
            p = line.split(",")
            if len(p) < 5:
                continue
            x, t, ty = float(p[0]), float(p[2]), int(p[3])
            col = min(max(int(x * 4 / 512), 0), 3)
            if ty & 128 and len(p) > 5:
                end = float(p[5].split(":")[0])
                if end > t:
                    notes.append((t, col, end))
                    continue
            notes.append((t, col, None))
    return keys, sorted(set(reds)), notes


def beat_times(reds, length_sec):
    """빨간 선들 → 박 시각(초). 첫 빨간 선 앞쪽은 같은 간격으로 0초까지 늘림."""
    reds = sorted(reds)
    out = []
    off0, mpb0, _ = reds[0]
    t = off0
    pre = []
    while t - mpb0 >= -mpb0 * 0.5 and t - mpb0 >= 0:
        t -= mpb0
        pre.append(t)
    out += sorted(pre)
    for i, (off, mpb, _m) in enumerate(reds):
        stop = reds[i + 1][0] if i + 1 < len(reds) else length_sec * 1000 + mpb
        k = 0
        while off + k * mpb < stop - 1e-6:
            out.append(off + k * mpb)
            k += 1
    b = np.array(out) / 1000.0
    return b[b <= length_sec + 60 / 30]  # 곡 끝 근처까지 (박 하나 여유)


def quantize(t_sec, beats):
    """시각 → 박 칸 위치(박 번호×48 + 칸) 와 오차(초)"""
    k = int(np.searchsorted(beats, t_sec, side="right") - 1)
    k = min(max(k, 0), len(beats) - 2)
    span = beats[k + 1] - beats[k]
    sub = int(round((t_sec - beats[k]) / span * SUB))
    pos = k * SUB + sub
    qt = beats[k] + sub / SUB * span
    return pos, t_sec - qt


# ── 조각 처리 ──────────────────────────────────────────────────────────────
def process_shard(shard: int, out: Path, eval_ratio: float):
    tar_path = out / f"tmp_{shard:06d}.tar"
    if not tar_path.exists():
        subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(tar_path), URL.format(shard)], check=True)
    stats = {"tracks": 0, "songs": 0, "charts": 0, "skipped_timing": 0, "q_err_ms": []}
    tf = tarfile.open(tar_path)
    members = {m.name: m for m in tf.getmembers()}
    for name, m in members.items():
        if not name.endswith(".json"):
            continue
        stats["tracks"] += 1
        d = json.load(tf.extractfile(m))
        if float(d.get("audio_length") or 0) > 600:  # 10분 넘는 곡은 건너뜀 (메모리 — 1시간짜리 곡에서 14GB 로 죽음)
            stats.setdefault("skipped_long", 0)
            stats["skipped_long"] += 1
            continue
        charts = []
        for b in d["beatmaps"]:
            c = b.get("content", "")
            if not re.search(r"^Mode:\s*3", c, re.M):
                continue
            keys, reds, notes = parse_osu(c)
            if keys is None or round(keys) != 4 or not reds or len(notes) < 20:
                continue
            # 비정상 박자선 거르기: BPM 30~400 밖이면 박이 수백만 개로 늘어 메모리가 터진다 (실제로 14GB 로 죽음)
            if any(not (150 <= bl <= 2000) for _t, bl, _m in reds):
                stats.setdefault("skipped_weird_bpm", 0)
                stats["skipped_weird_bpm"] += 1
                continue
            charts.append((b, reds, notes))
        if not charts:
            continue
        # 같은 곡의 악보들이 같은 박자선을 쓰는지 (다르면 가장 노트 많은 악보의 박자선만)
        main_reds = max(charts, key=lambda x: len(x[2]))[1]
        same = [ch for ch in charts if ch[1] == main_reds]
        stats["skipped_timing"] += len(charts) - len(same)
        audio_name = name[:-5] + ".opus"
        if audio_name not in members:
            continue
        raw = tf.extractfile(members[audio_name]).read()
        try:
            y = decode(raw)
        except subprocess.CalledProcessError:
            continue
        length = len(y) / SR
        if length > 600:
            continue
        beats = beat_times(main_reds, length)
        if len(beats) < 8 or len(beats) > length * 8:  # 초당 8박(480 BPM) 넘으면 비정상
            continue
        centers = (beats[:-1, None] + (beats[1:] - beats[:-1])[:, None] * (np.arange(SUB)[None, :] / SUB)).reshape(-1)
        mel = beat_aligned_mel(y, centers).astype(np.float16)

        chart_out = []
        for b, _reds, notes in same:
            ev = []
            for t, col, end in notes:
                pos, err = quantize(t / 1000, beats)
                stats["q_err_ms"].append(abs(err) * 1000)
                if end is None:
                    ev.append([pos, col, 1])
                else:
                    pe, err2 = quantize(end / 1000, beats)
                    stats["q_err_ms"].append(abs(err2) * 1000)
                    ev.append([pos, col, 2])
                    ev.append([max(pe, pos + 1), col, 3])
            ev.sort()
            try:
                star = float(b.get("difficultyrating") or 0)
            except ValueError:
                star = 0.0
            chart_out.append({"star": star, "version": b.get("version", ""), "beatmap_id": b.get("beatmap_id"),
                              "approved": b.get("approved"), "notes": ev})
        h = d.get("audio_hash", hashlib.sha256(raw).hexdigest())
        is_eval = int(h[:8], 16) % 1000 < eval_ratio * 1000
        meta = {"title": charts[0][0].get("title", ""), "artist": charts[0][0].get("artist", ""), "length": length,
                "bpm": charts[0][0].get("bpm"), "reds": main_reds, "shard": shard, "eval": is_eval, "hash": h}
        np.savez_compressed(out / "songs" / f"{h[:16]}.npz", mel=mel, beats=beats.astype(np.float64),
                            charts=json.dumps(chart_out), meta=json.dumps(meta))
        if is_eval:
            (out / "audio" / f"{h[:16]}.opus").write_bytes(raw)
        stats["songs"] += 1
        stats["charts"] += len(chart_out)
    tf.close()
    tar_path.unlink()
    q = np.array(stats.pop("q_err_ms")) if stats.get("q_err_ms") is not None else np.array([])
    stats["q_err_ms_median"] = round(float(np.median(q)), 2) if len(q) else None
    stats["q_err_ms_p99"] = round(float(np.percentile(q, 99)), 2) if len(q) else None
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-ratio", type=float, default=0.05)
    a = ap.parse_args()
    out = Path(a.out)
    for sub in ("songs", "audio"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log = out / "extract_log.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(l)["shard"] for l in log.read_text().splitlines() if l.strip()}
    for s in a.shards:
        if s in done:
            print(f"조각 {s}: 이미 처리함")
            continue
        t = time.perf_counter()
        st = process_shard(s, out, a.eval_ratio)
        st["shard"] = s
        st["seconds"] = round(time.perf_counter() - t, 1)
        with open(log, "a") as f:
            f.write(json.dumps(st) + "\n")
        n = len(list((out / "songs").glob("*.npz")))
        print(f"조각 {s}: {st} → 누적 곡 {n}개", flush=True)


if __name__ == "__main__":
    main()
