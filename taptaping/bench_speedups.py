"""속도 개선 실측: ① SV 생성 빼기 ② 모델·박자·오디오 분석을 4개 난이도가 공유 ③ 창 겹침 줄이기.

원본 inference.generate() 와 같은 순서로 돌리되
  - 박자(TIMING)는 공용 모델로 한 번만 만들어 메모리로 각 난이도에 넘긴다 (원본과 같은 extra_in_context 경로)
  - 노트는 MAP 만 만든다 (SV 생략)
  - 같은 모델·같은 오디오 창의 인코더 출력은 재사용한다
  - 박자용 / 노트용 창 겹침(lookback, lookahead)을 따로 줄 수 있다

  PYTHONPATH=. python taptaping/bench_speedups.py --audio song.mp3 --out out_dir \
      --timing-window 0.5,0.4 --map-window 0.25,0.25 --seed 7
"""
from __future__ import annotations

import argparse
import json
import time
from functools import reduce
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import inference as inf
import osuT5.osuT5.inference.processor as processor_mod
from osuT5.osuT5.dataset.data_utils import TIMING_TYPES, events_of_type, merge_events
from osuT5.osuT5.inference import Postprocessor, Preprocessor, Processor
from osuT5.osuT5.tokenizer import ContextType

REPO = Path(__file__).resolve().parents[1]

# ── 인코더 출력 재사용 (같은 모델 + 같은 오디오 창 + 같은 조건이면 그대로) ─────────────────
_orig_precompute = processor_mod.precompute_encoder_outputs
_enc_cache: dict = {}
ENC_STATS = {"computed": 0, "reused": 0, "seconds": 0.0}


def _cached_precompute(model, all_frames, cond_kwargs, song_positions, **kw):
    key = (id(model), tuple(all_frames.shape), float(all_frames.sum()), float(all_frames[..., ::997].sum()),
           tuple((k, float(v.float().sum())) for k, v in sorted(cond_kwargs.items())),
           None if song_positions is None else float(song_positions.sum()))
    if key in _enc_cache:
        ENC_STATS["reused"] += 1
        return _enc_cache[key]
    t = time.perf_counter()
    out = _orig_precompute(model, all_frames, cond_kwargs, song_positions, **kw)
    ENC_STATS["seconds"] += time.perf_counter() - t
    ENC_STATS["computed"] += 1
    _enc_cache[key] = out
    return out


processor_mod.precompute_encoder_outputs = _cached_precompute


def make_args(overrides: list[str]):
    with initialize_config_dir(config_dir=str(REPO / "configs" / "inference"), version_base="1.1"):
        cfg = compose(config_name="v32-mini", overrides=overrides)
    args = OmegaConf.to_object(cfg)
    inf.compile_args(args, verbose=False)
    return args


def gen_timing(args, model, tok):
    """원본 generate() 의 '박자 자동 생성' 분기와 같은 호출"""
    gcfg, _ = inf.get_config(args)
    pre = Preprocessor(args, parallel=args.parallel)
    seq = pre.segment(pre.load(args.audio_path))
    ev, tt = Processor(args, model, tok).generate(
        sequences=seq, generation_config=gcfg, in_context=[ContextType.NONE], out_context=[ContextType.TIMING],
        beatmap_path=args.beatmap_path, verbose=False)[0]
    ev, tt = events_of_type(ev, tt, TIMING_TYPES)
    return Postprocessor(args).generate_timing(ev), len(seq[0])


def gen_map(args, model, tok, timing, out_dir: Path):
    """원본 generate() 의 본 생성 분기와 같은 호출 (out_context = MAP 만)"""
    gcfg, bcfg = inf.get_config(args)
    pre = Preprocessor(args, parallel=args.parallel)
    seq = pre.segment(pre.load(args.audio_path))
    post = Postprocessor(args)
    result = Processor(args, model, tok).generate(
        sequences=seq, generation_config=gcfg, in_context=args.in_context, out_context=[ContextType.MAP],
        beatmap_path=args.beatmap_path, extra_in_context={ContextType.TIMING: timing}, verbose=False)
    events, _ = reduce(merge_events, result)
    if args.resnap_events:
        events = post.resnap_events(events, timing)
    res = post.generate(events=events, beatmap_config=bcfg, timing=timing)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "beatmap.osu"
    post.write_result(path, res)
    return path, len(seq[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--difficulties", default="1.5,2.5,3.5,4.5")
    ap.add_argument("--timing-window", default="0.5,0.4", help="박자 생성 lookback,lookahead")
    ap.add_argument("--map-window", default="0.5,0.4", help="노트 생성 lookback,lookahead")
    ap.add_argument("--hold-note-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tlb, tla = a.timing_window.split(",")
    mlb, mla = a.map_window.split(",")
    common = [f"audio_path={a.audio}", f"output_path={out}", "gamemode=3", "keycount=4", "device=cpu",
              "title=Track01", "artist=taptaping", f"hold_note_ratio={a.hold_note_ratio}", f"seed={a.seed}"]
    times: dict = {"map": {}, "windows": {}}

    t = time.perf_counter()
    base = make_args(common + ["difficulty=2.5"])
    load = dict(max_batch_size=base.max_batch_size, precision=base.precision,
                attn_implementation=base.attn_implementation, gamemode=3, fast_decoder_loop=base.fast_decoder_loop)
    model, tok = inf.load_model_with_server(base.model_path, base.train, base.device, auto_select_gamemode_model=True, **load)
    tmodel, ttok = inf.load_model_with_server(base.model_path, base.train, base.device, auto_select_gamemode_model=False, **load)
    times["load"] = time.perf_counter() - t

    # 박자: 한 번만 (원본에서 기본으로 쓰는 난이도 정보 없이도 되지만, 원본과 맞추려고 중간 난이도 2.5 로)
    t = time.perf_counter()
    targs = make_args(common + ["difficulty=2.5", f"lookback={tlb}", f"lookahead={tla}"])
    inf.setup_inference_environment(a.seed)
    timing, times["windows"]["timing"] = gen_timing(targs, tmodel, ttok)
    times["timing"] = time.perf_counter() - t

    for d in a.difficulties.split(","):
        t = time.perf_counter()
        margs = make_args(common + [f"difficulty={d}", f"lookback={mlb}", f"lookahead={mla}"])
        inf.setup_inference_environment(a.seed)
        path, times["windows"]["map"] = gen_map(margs, model, tok, timing, out / f"d{d}")
        times["map"][d] = time.perf_counter() - t
        print(f"  난이도 {d}: {times['map'][d]:.1f}초 → {path}", flush=True)

    times["total"] = times["load"] + times["timing"] + sum(times["map"].values())
    times["encoder"] = dict(ENC_STATS)
    times["settings"] = {"timing_window": a.timing_window, "map_window": a.map_window, "seed": a.seed,
                         "threads": torch.get_num_threads()}
    (out / "times.json").write_text(json.dumps(times, indent=2, ensure_ascii=False))
    print(json.dumps(times, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
