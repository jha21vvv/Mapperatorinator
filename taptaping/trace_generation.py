"""원본 생성 과정을 창(window) 단위로 기록한다 — 독립 구현(ONNX/C#)을 검증하기 위한 기준 자료.

기록 내용 (창마다): 문맥 종류, frame_time, 프롬프트 토큰, 생성된 토큰, lookback/lookahead, 결과 이벤트 수
그리고 박자(timing) 결과 이벤트, 최종 노트 이벤트, 토크나이저 표.

  PYTHONPATH=. python taptaping/trace_generation.py --audio clip.mp3 --out trace.json --difficulty 2.5
"""
from __future__ import annotations

import argparse
import json
import time
from functools import reduce
from pathlib import Path

import inference as inf
from osuT5.osuT5.dataset.data_utils import TIMING_TYPES, events_of_type, merge_events
from osuT5.osuT5.event import EventType
from osuT5.osuT5.inference import Postprocessor, Preprocessor, Processor
from osuT5.osuT5.tokenizer import ContextType

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_speedups import make_args  # noqa: E402

WINDOWS: list[dict] = []
_orig_generate = Processor.model_generate
_orig_add = Processor.add_predicted_tokens_to_context


def traced_generate(self, model_kwargs, **kw):
    result, stats = _orig_generate(self, model_kwargs, **kw)
    prompt = model_kwargs["decoder_input_ids"][0].tolist()
    WINDOWS.append({
        "context": kw.get("context_type"),
        "lookback_ms": kw.get("lookback_time", 0),
        "lookahead_ms": kw.get("lookahead_time", 0),
        "prompt": prompt,
        "generated": result[0, len(prompt):].tolist(),
    })
    return result, stats


def traced_add(self, context, predicted_tokens, frame_time, trim_lookback=False, trim_lookahead=False):
    WINDOWS[-1]["frame_time"] = float(frame_time)
    WINDOWS[-1]["trim_lookback"] = bool(trim_lookback)
    WINDOWS[-1]["trim_lookahead"] = bool(trim_lookahead)
    out = _orig_add(self, context, predicted_tokens, frame_time, trim_lookback, trim_lookahead)
    WINDOWS[-1]["events_after"] = len(context["events"])
    return out


Processor.model_generate = traced_generate
Processor.add_predicted_tokens_to_context = traced_add


def ev_list(events, times):
    return [[e.type.value, int(e.value) if isinstance(e.value, (int, float)) else e.value, float(t)] for e, t in zip(events, times)]


def tokenizer_tables(tok):
    return {
        "offset": tok.offset,
        "event_start": {k.value: v for k, v in tok.event_start.items()},
        "event_end": {k.value: v for k, v in tok.event_end.items()},
        "event_range": {k.value: [r.min_value, r.max_value] for k, r in tok.event_range.items()},
        "context_sos": {k.value: v for k, v in tok.context_sos.items()},
        "context_eos": {k.value: v for k, v in tok.context_eos.items()},
        "vocab_size_out": tok.vocab_size_out,
        "vocab_size_in": tok.vocab_size_in,
        "num_diff_classes": tok.num_diff_classes,
        "max_difficulty": tok.max_difficulty,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--difficulty", type=float, default=2.5)
    ap.add_argument("--timing-window", default="0.5,0.4")
    ap.add_argument("--map-window", default="0.25,0.25")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    tlb, tla = a.timing_window.split(",")
    mlb, mla = a.map_window.split(",")
    out_dir = Path(a.out).parent
    common = [f"audio_path={a.audio}", f"output_path={out_dir}", "gamemode=3", "keycount=4", "device=cpu",
              "hold_note_ratio=0.15", f"seed={a.seed}", f"difficulty={a.difficulty}"]

    base = make_args(common)
    load = dict(max_batch_size=base.max_batch_size, precision=base.precision,
                attn_implementation=base.attn_implementation, gamemode=3, fast_decoder_loop=base.fast_decoder_loop)
    model, tok = inf.load_model_with_server(base.model_path, base.train, base.device, auto_select_gamemode_model=True, **load)
    tmodel, ttok = inf.load_model_with_server(base.model_path, base.train, base.device, auto_select_gamemode_model=False, **load)

    trace = {"settings": vars(a), "tokenizer_map": tokenizer_tables(tok), "tokenizer_timing": tokenizer_tables(ttok)}

    # 박자
    targs = make_args(common + [f"lookback={tlb}", f"lookahead={tla}"])
    inf.setup_inference_environment(a.seed)
    gcfg, _ = inf.get_config(targs)
    pre = Preprocessor(targs, parallel=False)
    seq = pre.segment(pre.load(targs.audio_path))
    trace["song_length_ms"] = float(seq[2])
    trace["timing_frame_times"] = [float(x) for x in seq[1]]
    t = time.perf_counter()
    ev, tt = Processor(targs, tmodel, ttok).generate(
        sequences=seq, generation_config=gcfg, in_context=[ContextType.NONE], out_context=[ContextType.TIMING],
        verbose=False)[0]
    trace["timing_raw_events"] = ev_list(ev, tt)
    ev, tt = events_of_type(ev, tt, TIMING_TYPES)
    trace["timing_events"] = ev_list(ev, tt)
    post = Postprocessor(targs)
    timing = post.generate_timing(ev)
    trace["timing_points"] = [[float(tp.offset.total_seconds() * 1000), float(tp.ms_per_beat), int(tp.meter)] for tp in timing]
    trace["timing_seconds"] = time.perf_counter() - t
    n_timing_windows = len(WINDOWS)

    # 노트
    margs = make_args(common + [f"lookback={mlb}", f"lookahead={mla}"])
    inf.setup_inference_environment(a.seed)
    gcfg, bcfg = inf.get_config(margs)
    pre = Preprocessor(margs, parallel=False)
    seq = pre.segment(pre.load(margs.audio_path))
    trace["map_frame_times"] = [float(x) for x in seq[1]]
    t = time.perf_counter()
    result = Processor(margs, model, tok).generate(
        sequences=seq, generation_config=gcfg, in_context=margs.in_context, out_context=[ContextType.MAP],
        extra_in_context={ContextType.TIMING: timing}, verbose=False)
    events, times = reduce(merge_events, result)
    trace["map_events"] = ev_list(events, times)
    if margs.resnap_events:
        events = post.resnap_events(events, timing)
    res = post.generate(events=events, beatmap_config=bcfg, timing=timing)
    osu_path = out_dir / "trace_map.osu"
    post.write_result(osu_path, res)
    trace["map_seconds"] = time.perf_counter() - t
    trace["osu_path"] = str(osu_path)

    for w in WINDOWS[:n_timing_windows]:
        w["stage"] = "timing"
    for w in WINDOWS[n_timing_windows:]:
        w["stage"] = "map"
    trace["windows"] = WINDOWS
    Path(a.out).write_text(json.dumps(trace))
    print(f"창 {len(WINDOWS)}개 (박자 {n_timing_windows}, 노트 {len(WINDOWS) - n_timing_windows}) → {a.out}")


if __name__ == "__main__":
    main()
