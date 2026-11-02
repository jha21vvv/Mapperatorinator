"""Mapperatorinator 4K mania 생성 — torch/원본 코드 없이 numpy + ONNX 만으로 (Unity C# 이식용 참조 구현).

원본 파이프라인(inference.generate → Processor → Postprocessor) 중 4K mania 에 필요한 부분만 같은 규칙으로 옮겼다.
C# 으로 그대로 옮길 수 있게 일부러 단순한 반복문으로 썼다.

흐름
  1. 오디오 → 16kHz 모노, 최대값 정규화 → 창(2047×128 샘플)으로 자르기
  2. 창마다 멜 스펙트로그램(torchaudio 와 같은 계산) → encoder.onnx → 교차 어텐션 K/V
  3. 박자 모델로 창마다 토큰 생성 → 박 이벤트 → 박자선(timing points) 계산
  4. 박자선을 박 이벤트로 다시 풀어 문맥으로 넣고, 노트 모델로 창마다 토큰 생성
  5. 토큰 → 이벤트 → 스냅 보정(resnap) → 노트(시간·레인·롱노트)

  python taptaping/mini/mapper_mini.py --audio song.mp3 --models exports --difficulty 2.5 -o chart.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# 토크나이저
# ═══════════════════════════════════════════════════════════════════════════


class Tok:
    def __init__(self, tables: dict):
        self.t = tables
        self.start = tables["event_start"]
        self.range = tables["event_range"]
        self.offset = tables["offset"]
        self.eos = tables["eos_id"]
        self.sos = tables["sos_id"]
        self.ctx_sos = tables["context_sos"]
        self.ctx_eos = tables["context_eos"]
        # 토큰 → (종류, 값) 역표
        self.decode_table: list[tuple[str, int] | None] = [None] * tables["vocab_size_in"]
        for typ in tables["event_order"]:
            lo, hi = self.range[typ]
            s = self.start[typ]
            for v in range(lo, hi + 1):
                self.decode_table[s + v - lo] = (typ, v)
        self.ts_start = self.start["t"]
        self.ts_end = self.ts_start + self.range["t"][1] - self.range["t"][0] + 1

    def encode(self, typ: str, value: int) -> int:
        lo, hi = self.range[typ]
        if not lo <= value <= hi:
            raise ValueError(f"{typ}={value} 범위 밖 [{lo},{hi}]")
        return self.start[typ] + value - lo

    def decode(self, token: int):
        return self.decode_table[token] if 0 <= token < len(self.decode_table) else None

    @staticmethod
    def ratio_to_value(ratio: float, resolution: int) -> int:
        if ratio <= 0:
            return -1
        if ratio >= 1:
            return resolution + 1
        return int(min(max(round(ratio * resolution), 0), resolution))  # numpy round == 짝수 반올림 (Python round 와 같음)


# ═══════════════════════════════════════════════════════════════════════════
# 오디오 · 멜
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 16000
HOP = 128
N_FFT = 1024
N_MELS = 128
F_MIN, F_MAX = 20.0, 8000.0
FRAMES = 2048                      # src_seq_len
SAMPLES_PER_SEQ = (FRAMES - 1) * HOP  # 262016
MS_PER_SEQ = SAMPLES_PER_SEQ * 1000 / SAMPLE_RATE  # 16376.0


def load_audio(path: str) -> np.ndarray:
    """원본과 같은 방식 (pydub/ffmpeg 로 16kHz 모노 → 최대값으로 나눔). C# 에서는 AudioClip → 리샘플."""
    from pydub import AudioSegment
    a = AudioSegment.from_file(path).set_frame_rate(SAMPLE_RATE).set_channels(1)
    x = np.array(a.get_array_of_samples()).astype(np.float32)
    peak = np.max(np.abs(x)) if x.size else 0
    return x / peak if peak > 0 else x


def segment(samples: np.ndarray, lookback: float, lookahead: float):
    stride = int(SAMPLES_PER_SEQ * (1 - lookback - lookahead))
    song_length = len(samples) / SAMPLE_RATE * 1000
    if len(samples) < SAMPLES_PER_SEQ:
        pad = SAMPLES_PER_SEQ - len(samples)
    else:
        rem = (len(samples) - SAMPLES_PER_SEQ) % stride
        pad = 0 if rem == 0 else stride - rem
    samples = np.pad(samples, (0, pad))
    n = (len(samples) - SAMPLES_PER_SEQ) // stride + 1
    ms_per_stride = stride * 1000 / SAMPLE_RATE
    # 원본: torch.arange(0, n*ms_per_stride, ms_per_stride).to(int32) — 소수점 버림
    times = [int(i * ms_per_stride) for i in range(n)]
    windows = [samples[i * stride: i * stride + SAMPLES_PER_SEQ] for i in range(n)]
    return windows, times, song_length


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """torchaudio.functional.melscale_fbanks(norm=None, mel_scale='htk') 와 같음 → [n_freqs, n_mels]"""
    n_freqs = N_FFT // 2 + 1
    all_freqs = np.linspace(0, SAMPLE_RATE // 2, n_freqs, dtype=np.float32)
    m_pts = np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2, dtype=np.float32)
    f_pts = _mel_to_hz(m_pts).astype(np.float32)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float32)


_FB = None
_WIN = None


def mel_spectrogram(window: np.ndarray) -> np.ndarray:
    """torchaudio MelSpectrogram(center=True, pad_mode='reflect', power=2, 주기형 Hann) + log1p → [128, 2048]"""
    global _FB, _WIN
    if _FB is None:
        _FB = mel_filterbank()
        _WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
    x = np.pad(window.astype(np.float32), (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * _WIN, axis=1)) ** 2  # [frames, 513]
    mel = spec.astype(np.float32) @ _FB  # [frames, 128]
    return np.log1p(mel).T.astype(np.float32)  # [128, frames]


# ═══════════════════════════════════════════════════════════════════════════
# 박자선 (Postprocessor.generate_timing / OsuParser.parse_timing / resnap)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TP:  # 빨간 선만 다룸 (offset ms, ms_per_beat, meter)
    offset: float
    mpb: float
    meter: int = 4


@dataclass
class Marker:
    time: float
    is_measure: bool
    is_redline: bool
    beats: float = 1  # beats_from_last_marker


def tp_at(time: float, timing: list[TP]) -> TP:
    for tp in reversed(timing):
        if tp.offset <= time:
            return tp
    return timing[0]


def tp_after(time: float, timing: list[TP]):
    for tp in timing:
        if tp.offset > time:
            return tp
    return None


def add_redline(timing: list[TP], new: TP, set_mpb=False, set_meter=False) -> list[TP]:
    """TimingPointsChange(uninherited=True).add_change(all_after=True) 의 빨간 선 전용 판"""
    prev = None
    on = []
    for tp in timing:
        if tp.offset < new.offset and (prev is None or tp.offset >= prev.offset):
            prev = tp
        if abs(tp.offset - new.offset) <= 2.0:  # fuzzyness 2ms (math.isclose abs_tol)
            on.append(tp)
    if on:
        prev = on[-1]
    adding = None
    if not on:
        adding = TP(new.offset, new.mpb, new.meter) if prev is None else TP(new.offset, prev.mpb, prev.meter)
        on.append(adding)
    for tp in on:
        if set_mpb:
            tp.mpb = new.mpb
        if set_meter:
            tp.meter = new.meter
    if adding is not None:
        timing.append(adding)
    timing.sort(key=lambda t: t.offset)
    return timing


def is_snapped(t, resnapped, leniency):
    return abs(t - resnapped) <= leniency


def get_ms_per_beat(time_from_redline, beats_from_redline, leniency):
    if beats_from_redline == 0 or time_from_redline == 0:
        return 1000
    mpb = time_from_redline / beats_from_redline
    bpm = 60000 / mpb
    if bpm < 1:
        return 1000
    for mul in (1, 2, 10, 100, 1000):
        cand = 60000 / (round(bpm * mul) / mul)
        if is_snapped(time_from_redline, cand * beats_from_redline, leniency):
            return cand
    return mpb


def check_ms_per_beat(mpb_new, markers, redline: TP, leniency):
    beats = 0
    off = round(redline.offset)
    for m in markers:
        beats += m.beats
        if not is_snapped(m.time, off + mpb_new * beats, leniency):
            return False
    return True


def human_round_ms_per_beat(mpb, markers, redline, leniency):
    if mpb == 0 or mpb > 60000:
        return mpb
    bpm = 60000 / mpb
    for mul in (1, 2, 10, 100, 1000):
        cand = 60000 / (round(bpm * mul) / mul)
        if check_ms_per_beat(cand, markers, redline, leniency):
            return cand
    return mpb


def generate_timing(events: list[tuple[str, int]], leniency: float) -> list[TP]:
    """박 이벤트(t, beat/measure/timing_point) → 빨간 선 목록 (원본 generate_timing, types_first=False)"""
    markers: list[Marker] = []
    for i, (typ, val) in enumerate(events):
        if typ in ("beat", "measure", "timing_point") and i - 1 >= 0 and events[i - 1][0] == "t":
            markers.append(Marker(int(events[i - 1][1]), typ == "measure", typ == "timing_point",
                                  0 if typ == "timing_point" else 1))
    if not markers:
        return []
    markers.sort(key=lambda m: m.time)
    timing: list[TP] = []
    for m in markers:
        if m.is_redline:
            timing = add_redline(timing, TP(m.time, 1000, 4))
    if not timing or timing[0].offset > markers[0].time:
        timing.insert(0, TP(markers[0].time, 1000, 4))

    counter = 0
    last_measure_time = markers[0].time
    for m in markers:
        t = m.time
        if m.is_redline:
            counter = 0
            last_measure_time = t
            continue
        red = tp_at(t - 1, timing)
        if round(red.offset) == t:
            continue
        counter += 1
        if not m.is_measure:
            continue
        if red.meter != counter:
            if last_measure_time <= round(red.offset):
                red.meter = counter
            else:
                timing = add_redline(timing, TP(last_measure_time, 1000, counter), set_meter=True)
        counter = 0
        last_measure_time = t

    counter = 0
    last_mpb = 1000
    for m in markers:
        t = m.time
        red = tp_at(t - 1, timing)
        red_off = round(red.offset)
        if red_off == t:
            counter = 0
            continue
        before = [o for o in markers if red_off + 1 <= o.time < t] + [m]
        if m.beats == 0:
            if len(before) != 1:
                counter = 0
                continue
            b = (t - red_off) / last_mpb
            cands = [round(b), 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]
            m.beats = min(cands, key=lambda x: abs(x - b))

        def get_mpb(ms, off):
            return get_ms_per_beat(ms[-1].time - off, sum(x.beats for x in ms), 0)

        mpb = get_mpb(before, red_off)
        if check_ms_per_beat(mpb, before, red, leniency):
            red.mpb = human_round_ms_per_beat(mpb, before, red, leniency)
            mpb = red.mpb
        elif len(before) > 1:
            def test_split(ms, off):
                mp = get_mpb(ms, off)
                loss, beats = 0.0, 0
                for x in ms:
                    beats += x.beats
                    loss += (x.time - (off + beats * mp)) ** 2
                return loss / len(ms)

            best_loss, best_split = math.inf, len(before) - 1
            for i in range(1, len(before)):
                split = before[i - 1]
                if not split.is_measure and i != len(before) - 1:
                    continue
                loss = test_split(before[:i], red_off) + test_split(before[i:], split.time)
                if loss < best_loss:
                    best_loss, best_split = loss, i
            if best_split < len(before) - 1:
                mp = get_mpb(before[:best_split], red_off)
                red.mpb = human_round_ms_per_beat(mp, before[:best_split], red, leniency)
            last_time = before[best_split - 1].time
            beats_from_split = sum(x.beats for x in before[best_split:])
            mpb = get_ms_per_beat(t - last_time, beats_from_split, leniency)
            timing = add_redline(timing, TP(last_time, mpb, 4), set_mpb=True)
            counter = 0
            for i in range(len(before) - 2, best_split - 1, -1):
                if before[i].is_measure:
                    break
                counter += 1
        last_mpb = mpb
        counter += 1
        red = tp_at(t, timing)
        if round(red.offset) == t:
            counter = 0
        if m.is_measure:
            if counter % red.meter != 0:
                timing = add_redline(timing, TP(t, red.mpb, red.meter), set_mpb=True)
            counter = 0
    return timing


def parse_timing(timing: list[TP], song_length: float):
    """박자선 → 박 이벤트 [(종류, 값)], 시각 (OsuParser.parse_timing, types_first=False, add_timing_points=True)"""
    events, times = [], []
    for i, tp in enumerate(timing):
        next_time = timing[i + 1].offset - 10 if i + 1 < len(timing) else song_length
        start = tp.offset
        t = start
        k = 0
        while t <= next_time:
            typ = "timing_point" if k == 0 else ("measure" if k % tp.meter == 0 else "beat")
            ms = int(t + 1e-5)
            events += [("t", ms), (typ, 0)]
            times += [ms, ms]
            k += 1
            t = int(start + k * tp.mpb)
    return events, times


IGNORE_TICKS = {1: [], 4: [2], 6: [2, 3], 8: [4], 9: [3], 10: [2, 5], 12: [4, 6], 14: [2, 7], 15: [3, 5], 16: [8]}


def resnap(t: float, timing: list[TP], divisor: int) -> float:
    if not timing:
        return t
    cur = tp_at(t, timing)
    cur_t = round(cur.offset)
    before = tp_at(cur_t - 1, timing)
    after = tp_after(t, timing)
    after_t = round(after.offset) if after is not None else math.inf

    def local_ticks(tp, div):
        tp_t = round(tp.offset)
        step = tp.mpb / div
        rem = (t - tp_t) % step  # Python % (음수도 양수 나머지)
        return {int(t - rem - step), int(t - rem), int(t - rem + step), int(t - rem + 2 * step)}

    def ticks_minus(tp, div):
        ticks = local_ticks(tp, div)
        for ig in IGNORE_TICKS.get(div, [1]):
            ticks -= local_ticks(tp, ig)
        return ticks

    m = 20
    ticks = {x for x in ticks_minus(cur, divisor) if cur_t - m <= x <= after_t + m}
    if before is not None:
        ticks |= {x for x in ticks_minus(before, divisor) if x <= cur_t + m}
    if after is not None:
        ticks |= {x for x in ticks_minus(after, divisor) if x >= after_t - m}
    if not ticks:
        return t
    return min(ticks, key=lambda x: abs(x - t))  # 같은 거리면 set 순회 순서 — C# 에서는 작은 값 우선으로 고정


# ═══════════════════════════════════════════════════════════════════════════
# ONNX 모델
# ═══════════════════════════════════════════════════════════════════════════


class OnnxModel:
    def __init__(self, folder: Path, threads: int = 0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.meta = json.loads((folder / "meta.json").read_text())
        self.enc = ort.InferenceSession(str(folder / "encoder.onnx"), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(folder / "decoder.onnx"), so, providers=["CPUExecutionProvider"])
        n = self.meta["layers"]
        self.kv_names = [f"{t}_{i}" for i in range(n) for t in ("k", "v")]
        self.H, self.D = self.meta["heads"], self.meta["head_dim"]
        self.stats = {"encoder_ms": 0.0, "decoder_ms": 0.0, "tokens": 0, "prefill_tokens": 0, "windows": 0}

    def encode(self, mel: np.ndarray):
        t = time.perf_counter()
        out = self.enc.run(None, {"mel": mel[None]})
        self.stats["encoder_ms"] += (time.perf_counter() - t) * 1000
        self.stats["windows"] += 1
        return out

    def start(self, cross, prompt: list[int]):
        feeds = {"input_ids": np.array([prompt], dtype=np.int64)}
        feeds.update({f"past_{s}": np.zeros((1, self.H, 0, self.D), np.float32) for s in self.kv_names})
        feeds.update({f"cross_{s}": c for s, c in zip(self.kv_names, cross)})
        return self._run(feeds, len(prompt))

    def step(self, feeds, token: int):
        feeds["input_ids"] = np.array([[token]], dtype=np.int64)
        return self._run(feeds, 1)

    def _run(self, feeds, n):
        t = time.perf_counter()
        r = self.dec.run(None, feeds)
        self.stats["decoder_ms"] += (time.perf_counter() - t) * 1000
        if n == 1:
            self.stats["tokens"] += 1
        else:
            self.stats["prefill_tokens"] += n
        feeds.update({f"past_{s}": p for s, p in zip(self.kv_names, r[1:])})
        return r[0][0].astype(np.float64), feeds


# ═══════════════════════════════════════════════════════════════════════════
# 생성 (Processor.generate_sequential + model_generate, 4K mania 전용)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Settings:
    difficulty: float = 2.5
    keycount: int = 4
    hold_note_ratio: float | None = 0.15
    hitsounded: bool = True
    temperature: float = 0.9
    top_p: float = 0.9
    timing_window: tuple[float, float] = (0.5, 0.4)
    map_window: tuple[float, float] = (0.25, 0.25)
    seed: int = 7


@dataclass
class Context:
    kind: str  # "timing" | "map"
    events: list = field(default_factory=list)  # [(종류, 값)]
    times: list = field(default_factory=list)
    finished: bool = False


class Generator:
    def __init__(self, tables: dict, timing_model: OnnxModel, map_model: OnnxModel, s: Settings):
        self.tok = Tok(tables)
        self.tables = tables
        self.timing_model, self.map_model, self.s = timing_model, map_model, s
        self.rng = np.random.default_rng(s.seed)
        self.trace: list[dict] = []

    # ── 프롬프트 ───────────────────────────────────────────────────────
    def class_tokens(self, song_length: float) -> list[int]:
        k = self.tok
        nd, md = self.tables["num_diff_classes"], self.tables["max_difficulty"]
        diff = int(min(max(int(self.s.difficulty * nd / md), 0), nd - 1))
        hnr = (k.encode("hold_note_ratio", 12) if self.s.hold_note_ratio is None
               else k.encode("hold_note_ratio", Tok.ratio_to_value(self.s.hold_note_ratio, 10)))
        return [
            k.encode("gamemode", 3),
            k.encode("difficulty", diff),
            k.encode("mapper", self.tables["num_mapper_classes"]),  # 모름
            k.encode("year", 2077),  # 모름
            k.encode("hitsounded", int(self.s.hitsounded)),
            k.encode("song_length", int(min(max(song_length // 10000, 0), 60))),
            k.encode("keycount", self.s.keycount),
            hnr,
            k.encode("scroll_speed_ratio", 12),  # 모름
            k.encode("descriptor", self.tables["num_descriptor_classes"]),  # 모름
        ]

    def encode_events(self, events, frame_time) -> list[int]:
        lo, hi = self.tok.range["t"]
        out = []
        for typ, val in events:
            if typ == "t":
                v = int((val - frame_time) / 10)
                val = min(max(v, lo), hi)
            out.append(self.tok.encode(typ, val))
        return out

    @staticmethod
    def time_range(times, start, end):
        s = 0
        for i in range(len(times) - 1, -1, -1):
            if times[i] < start:
                s = i + 1
                break
        e = 0
        for i in range(len(times) - 1, -1, -1):
            if times[i] < end:
                e = i + 1
                break
        return s, e

    def last_sv_token(self, contexts, frame_time) -> int:
        for c in contexts:
            if c.kind == "map":
                for i in range(len(c.events) - 1, -1, -1):
                    if c.events[i][0] == "scroll_speed" and c.times[i] < frame_time:
                        return self.tok.encode("scroll_speed", c.events[i][1])
        return self.tok.encode("scroll_speed", 100)

    def build_prompt(self, cls, contexts: list[Context], frame_time, song_length) -> list[int]:
        k = self.tok
        prompt = list(cls)
        prompt.append(self.last_sv_token(contexts, frame_time))
        prompt.append(k.encode("song_position", Tok.ratio_to_value(frame_time / song_length, 100)))
        prompt.append(k.sos)
        for i, c in enumerate(contexts):
            s, e = self.time_range(c.times, frame_time, frame_time + MS_PER_SEQ)
            prompt.append(k.ctx_sos[c.kind])
            prompt += self.encode_events(c.events[s:e], frame_time)
            if i != len(contexts) - 1:
                prompt.append(k.ctx_eos[c.kind])
        if len(prompt) >= self.tables["tgt_seq_len"]:
            raise ValueError("프롬프트가 너무 깁니다 (원본은 문맥을 반으로 줄여 재시도 — 4K 에선 드묾)")
        return prompt

    # ── 토큰 고르기 ────────────────────────────────────────────────────
    def sample(self, logits, seq: list[int], lookback_end: int | None) -> int:
        k = self.tok
        x = logits.copy()
        # 1) 시간 역행 금지: 마지막 SOS 이후의 마지막 시간 토큰보다 작은 시간 금지
        sos_ids = {k.sos, *k.ctx_sos.values()}
        for tkn in reversed(seq):
            if tkn in sos_ids:
                break
            if k.ts_start <= tkn < k.ts_end:
                x[k.ts_start:tkn] = -np.inf
                break
        # 2) 온도
        x = x / self.s.temperature
        # 3) 앞 구간(lookback) 시간 금지 — 질량은 허용 시간으로 옮김
        if lookback_end is not None:
            tm = _logsumexp(x[k.ts_start:k.ts_end])
            am = _logsumexp(x[lookback_end:k.ts_end])
            if np.isfinite(am):
                x[lookback_end:k.ts_end] += tm - am
            else:
                x[k.eos] = np.logaddexp(x[k.eos], tm)
            x[k.ts_start:lookback_end] = -np.inf
        # 4) top-p
        order = np.argsort(x, kind="stable")  # 오름차순
        sx = x[order]
        p = np.exp(sx - sx[-1])
        p /= p.sum()
        cum = np.cumsum(p)
        remove = cum <= (1 - self.s.top_p)
        remove[-1] = False
        x[order[remove]] = -np.inf
        # 5) 뽑기
        p = np.exp(x - np.max(x))
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))

    # ── 창 하나 ───────────────────────────────────────────────────────
    def run_window(self, model: OnnxModel, cross, prompt, kind, trim_lb, trim_la, lb_ms, la_ms):
        k = self.tok
        eos = {k.eos, k.ctx_eos[kind]}
        lookback_end = None
        if trim_lb and lb_ms > 0:
            n = int(lb_ms / 10)
            eos |= set(range(k.ts_start, k.ts_start + n))
            lookback_end = k.encode("t", n)
        if trim_la and la_ms > 0:
            eos |= set(range(k.ts_end - int(la_ms / 10), k.ts_end))
        max_new = self.tables["tgt_seq_len"] - len(prompt)
        logits, feeds = model.start(cross, prompt)
        seq = list(prompt)
        gen = []
        for _ in range(max_new):
            tkn = self.sample(logits, seq, lookback_end)
            gen.append(tkn)
            seq.append(tkn)
            if tkn in eos:
                break
            logits, feeds = model.step(feeds, tkn)
        return gen

    def add_tokens(self, c: Context, tokens, frame_time, trim_lb, trim_la, lookback, lookahead):
        k = self.tok
        while tokens and (tokens[-1] == k.eos or tokens[-1] == k.ctx_eos[c.kind]):
            tokens = tokens[:-1]
        lb_range_end = k.encode("t", int(lookback * MS_PER_SEQ / 10))
        la_range_start = k.encode("t", int((1 - lookahead) * MS_PER_SEQ / 10))
        if tokens and ((trim_la and la_range_start <= tokens[-1] < k.ts_end) or
                       (trim_lb and k.ts_start <= tokens[-1] < lb_range_end)):
            tokens = tokens[:-1]
        for tkn in tokens:
            if tkn == k.eos:
                break
            d = k.decode(tkn)
            if d is None:
                continue
            typ, val = d
            if typ == "t":
                val = frame_time + val * 10 + (5 if val >= 0 else 0)
            c.events.append((typ, val))
        # 시각: 직전 시간 토큰 값 (mania 에는 보간 대상 이벤트 없음)
        cur = c.times[-1] if c.times else 0
        for i in range(len(c.times), len(c.events)):
            if c.events[i][0] == "t":
                cur = c.events[i][1]
            c.times.append(cur)
        if trim_la:
            limit = frame_time + (1 - lookahead) * MS_PER_SEQ
            while c.times and c.times[-1] > limit:
                c.events.pop()
                c.times.pop()

    def generate_stage(self, model, kind, contexts, windows, frame_times, song_length, lookback, lookahead):
        cls = self.class_tokens(song_length)
        out = contexts[-1]
        lb_ms, la_ms = lookback * MS_PER_SEQ, lookahead * MS_PER_SEQ
        for wi, (w, ft) in enumerate(zip(windows, frame_times)):
            trim_lb = wi != 0 and lookback > 0
            trim_la = wi != len(windows) - 1
            cross = model.encode(mel_spectrogram(w))
            prompt = self.build_prompt(cls, contexts, ft, song_length)
            gen = self.run_window(model, cross, prompt, kind, trim_lb, trim_la, lb_ms, la_ms)
            self.trace.append({"stage": kind, "frame_time": ft, "prompt": prompt, "generated": gen})
            self.add_tokens(out, gen, ft, trim_lb, trim_la, lookback, lookahead)

    # ── 전체 ──────────────────────────────────────────────────────────
    def generate_timing_stage(self, samples: np.ndarray):
        tw, ts, song_length = segment(samples, *self.s.timing_window)
        timing_ctx = Context("timing")
        self.generate_stage(self.timing_model, "timing", [timing_ctx], tw, ts, song_length, *self.s.timing_window)
        beat_events = [e for e in timing_ctx.events if e[0] in ("t", "beat", "measure", "timing_point")]
        return generate_timing(beat_events, self.tables["timing_leniency"]), song_length

    def generate_map_stage(self, samples: np.ndarray, timing: list[TP], song_length: float, difficulty: float):
        """박자선이 주어졌을 때 난이도 하나의 노트 생성. 인코더 결과는 메모리 절약을 위해 난이도마다 다시 계산."""
        self.s.difficulty = difficulty
        tev, ttimes = parse_timing(timing, song_length)
        given = Context("timing", tev, ttimes, finished=True)
        map_ctx = Context("map")
        mw, mt, _ = segment(samples, *self.s.map_window)
        self.generate_stage(self.map_model, "map", [given, map_ctx], mw, mt, song_length, *self.s.map_window)
        return events_to_notes(map_ctx.events, timing, self.s.keycount)

    def generate(self, samples: np.ndarray, difficulties=None):
        difficulties = difficulties or [self.s.difficulty]
        t0 = time.perf_counter()
        timing, song_length = self.generate_timing_stage(samples)
        sec = {"timing": time.perf_counter() - t0}
        notes = {}
        for d in difficulties:
            t = time.perf_counter()
            notes[d] = self.generate_map_stage(samples, timing, song_length, d)
            sec[f"map_{d}"] = time.perf_counter() - t
        return {"timing": timing, "notes": notes, "song_length": song_length, "seconds": sec}


def _logsumexp(v):
    m = np.max(v)
    if not np.isfinite(m):
        return -np.inf
    return m + np.log(np.sum(np.exp(v - m)))


# ═══════════════════════════════════════════════════════════════════════════
# 이벤트 → 노트
# ═══════════════════════════════════════════════════════════════════════════


TYPE_EVENTS = {"circle", "spinner", "spinner_end", "slider_head", "bezier_anchor", "perfect_anchor", "catmull_anchor",
               "red_anchor", "last_anchor", "slider_end", "beat", "measure", "timing_point", "kiai", "hold_note",
               "hold_note_end", "drumroll", "drumroll_end", "denden", "denden_end", "scroll_speed_change"}


def events_to_notes(events, timing: list[TP], keycount: int):
    """resnap_events + 그룹 묶기 + 롱노트 짝짓기 (원본 Postprocessor, mania). 결과: [시작ms, 레인1~4, 끝ms 또는 None]"""
    ev = list(events)
    for i, (typ, val) in enumerate(ev):  # 스냅 보정
        if typ == "t" and i + 1 < len(ev) and ev[i + 1][0] == "snap" and ev[i + 1][1] > 0:
            ev[i] = ("t", int(resnap(val, timing, ev[i + 1][1])))
    notes, holds = [], {}
    g_time, g_x = None, None
    last_x = 256  # 원본: 위치 토큰이 없으면 직전 위치(처음엔 256)
    for typ, val in ev:
        if typ == "t":
            g_time = val
        elif typ == "column":
            g_x = int((val + 0.5) * 512 / keycount)  # 원본 _convert_column_to_position
        elif typ in TYPE_EVENTS:
            if g_time is not None:  # 시간 없는 그룹은 버림
                x = last_x if g_x is None else g_x
                if typ == "circle":
                    last_x = x
                col = min(max(int(x * keycount / 512), 0), keycount - 1)
                t = int(round(g_time))
                if typ == "circle":
                    notes.append([t, col + 1, None])
                elif typ == "hold_note":
                    holds[col] = t
                elif typ == "hold_note_end":
                    st = holds.pop(col, None)
                    if st is not None:
                        notes.append([st, col + 1, t if t > st else None])  # 길이 0 롱노트는 단타
            g_time, g_x = None, None
    notes.sort(key=lambda n: n[0])
    return notes


def notes_to_taptaping(notes) -> list[dict]:
    out = []
    for t, lane, end in notes:
        if end is not None and end > t:
            out.append({"time": round(t / 1000, 4), "lane": lane, "type": 1, "duration": round((end - t) / 1000, 4), "isSlide": 0})
        else:
            out.append({"time": round(t / 1000, 4), "lane": lane, "type": 0, "duration": 0.0, "isSlide": 0})
    return out


def write_osu(path: str, timing: list[TP], notes, keycount: int = 4):
    """비교·확인용 최소 .osu (mania)"""
    lines = ["osu file format v14", "", "[General]", "AudioFilename: audio.mp3", "Mode: 3", "", "[Metadata]",
             "Title:AI", "Creator:Mapperatorinator (mini)", "Version:mini", "", "[Difficulty]",
             f"CircleSize:{keycount}", "", "[TimingPoints]"]
    lines += [f"{tp.offset:g},{tp.mpb!r},{tp.meter},2,0,100,1,0" for tp in timing]
    lines += ["", "[HitObjects]"]
    for t, lane, end in notes:
        x = int((lane - 0.5) * 512 / keycount)
        lines.append(f"{x},192,{t},128,0,{end}:0:0:0:0:" if end is not None else f"{x},192,{t},1,0,0:0:0:0:")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--models", required=True, help="timing/, map/, mapper_tables.json 이 있는 폴더")
    ap.add_argument("--difficulties", default="2.5", help="쉼표로 여러 개 (예: 1.5,2.5,3.5,4.5 → easy~insane)")
    ap.add_argument("--map-window", default="0.25,0.25")
    ap.add_argument("--timing-window", default="0.5,0.4")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("-o", "--output", required=True, help="결과 JSON")
    ap.add_argument("--osu-dir", help="난이도별 비교용 .osu 를 저장할 폴더 (d{난이도}/beatmap.osu)")
    a = ap.parse_args()
    root = Path(a.models)
    tables = json.loads((root / "mapper_tables.json").read_text())
    diffs = [float(x) for x in a.difficulties.split(",")]
    s = Settings(seed=a.seed, map_window=tuple(float(x) for x in a.map_window.split(",")),
                 timing_window=tuple(float(x) for x in a.timing_window.split(",")))
    g = Generator(tables, OnnxModel(root / "timing", a.threads), OnnxModel(root / "map", a.threads), s)
    res = g.generate(load_audio(a.audio), diffs)
    out = {"timing": [[tp.offset, tp.mpb, tp.meter] for tp in res["timing"]],
           "notes": {str(d): notes_to_taptaping(n) for d, n in res["notes"].items()},
           "seconds": res["seconds"], "stats": {"timing": g.timing_model.stats, "map": g.map_model.stats}}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=1))
    if a.osu_dir:
        for d, n in res["notes"].items():
            (Path(a.osu_dir) / f"d{d}").mkdir(parents=True, exist_ok=True)
            write_osu(str(Path(a.osu_dir) / f"d{d}" / "beatmap.osu"), res["timing"], n)
    print(f"박자선 {len(res['timing'])}개 · " + " · ".join(f"{d}: 노트 {len(n)}개" for d, n in res["notes"].items()))
    print("초: " + json.dumps({k: round(v, 1) for k, v in res["seconds"].items()}))
    print(json.dumps(out["stats"]))


if __name__ == "__main__":
    main()
