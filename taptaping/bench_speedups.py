"""속도 개선 실측: ① SV 생성 빼기 ② 모델·박자·오디오 분석을 4개 난이도가 공유 ③ 창 겹침 줄이기.

기준(원본 inference.py 를 난이도마다 따로 실행)과 같은 곡·같은 시드로 4개 난이도를 만들고 단계별 시간을 잰다.

  PYTHONPATH=. python taptaping/bench_speedups.py --audio song.mp3 --out out_dir --lookback 0.5 --lookahead 0.4
  (lookback/lookahead 를 원본 기본값으로 두면 ①② 만, 줄이면 ①②③)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import inference as inf
import osuT5.osuT5.inference.processor as processor_mod
from osuT5.osuT5.event import ContextType

REPO = Path(__file__).resolve().parents[1]

# ── ② 오디오 분석(인코더) 결과를 같은 모델·같은 오디오면 재사용 ─────────────────────────
_orig_precompute = processor_mod.precompute_encoder_outputs
_enc_cache: dict = {}
ENC_STATS = {"computed": 0, "reused": 0}


def _cached_precompute(model, all_frames, cond_kwargs, song_positions, **kw):
    key = (id(model), tuple(all_frames.shape), float(all_frames.sum()), float(all_frames[..., ::997].sum()),
           tuple((k, float(v.float().sum())) for k, v in sorted(cond_kwargs.items())),
           None if song_positions is None else float(song_positions.sum()))
    if key in _enc_cache:
        ENC_STATS["reused"] += 1
        return _enc_cache[key]
    ENC_STATS["computed"] += 1
    out = _orig_precompute(model, all_frames, cond_kwargs, song_positions, **kw)
    _enc_cache[key] = out
    return out


processor_mod.precompute_encoder_outputs = _cached_precompute


def make_args(overrides: list[str]):
    with initialize_config_dir(config_dir=str(REPO / "configs" / "inference"), version_base="1.1"):
        cfg = compose(config_name="v32-mini", overrides=overrides)
    args = OmegaConf.to_object(cfg)
    inf.compile_args(args, verbose=False)
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--difficulties", default="1.5,2.5,3.5,4.5")
    ap.add_argument("--lookback", type=float, default=0.5)
    ap.add_argument("--lookahead", type=float, default=0.4)
    ap.add_argument("--hold-note-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    common = [f"audio_path={a.audio}", "gamemode=3", "keycount=4", "device=cpu",
              f"lookback={a.lookback}", f"lookahead={a.lookahead}", "title=Track01", "artist=taptaping"]
    times: dict = {}

    # 모델 2개를 한 번만 불러온다
    t = time.perf_counter()
    base = make_args(common + [f"output_path={out}", "difficulty=2.5"])
    model, tok = inf.load_model_with_server(base.model_path, base.train, base.device, max_batch_size=base.max_batch_size,
                                            precision=base.precision, attn_implementation=base.attn_implementation,
                                            gamemode=3, auto_select_gamemode_model=True,
                                            fast_decoder_loop=base.fast_decoder_loop)
    tmodel, ttok = inf.load_model_with_server(base.model_path, base.train, base.device, max_batch_size=base.max_batch_size,
                                              precision=base.precision, attn_implementation=base.attn_implementation,
                                              gamemode=3, auto_select_gamemode_model=False,
                                              fast_decoder_loop=base.fast_decoder_loop)
    times["load"] = time.perf_counter() - t

    # 박자를 한 번만 만든다 (공용 모델, 출력 = TIMING 만)
    t = time.perf_counter()
    targs = make_args(common + [f"output_path={out / 'timing'}", "difficulty=2.5", "output_type=[TIMING]", f"seed={a.seed}"])
    (out / "timing").mkdir(exist_ok=True)
    inf.setup_inference_environment(a.seed)
    gcfg, bcfg = inf.get_config(targs)
    _, timing_path = inf.generate(targs, generation_config=gcfg, beatmap_config=bcfg, model=tmodel, tokenizer=ttok,
                                  verbose=False)
    times["timing"] = time.perf_counter() - t

    # 난이도별 노트만 생성 (① SV 없음, 박자는 위 결과를 문맥으로)
    times["map"] = {}
    for d in a.difficulties.split(","):
        t = time.perf_counter()
        margs = make_args(common + [f"output_path={out / ('d' + d)}", f"difficulty={d}", f"hold_note_ratio={a.hold_note_ratio}",
                                    f"beatmap_path='{timing_path}'", "in_context=[TIMING]", "output_type=[MAP]",
                                    f"seed={a.seed}"])
        (out / ("d" + d)).mkdir(exist_ok=True)
        inf.setup_inference_environment(a.seed)
        gcfg, bcfg = inf.get_config(margs)
        inf.generate(margs, generation_config=gcfg, beatmap_config=bcfg, model=model, tokenizer=tok, verbose=False)
        times["map"][d] = time.perf_counter() - t
        print(f"  난이도 {d}: {times['map'][d]:.1f}초", flush=True)

    times["total"] = times["load"] + times["timing"] + sum(times["map"].values())
    times["encoder"] = dict(ENC_STATS)
    times["settings"] = {"lookback": a.lookback, "lookahead": a.lookahead, "threads": torch.get_num_threads()}
    (out / "times.json").write_text(json.dumps(times, indent=2, ensure_ascii=False))
    print(json.dumps(times, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
