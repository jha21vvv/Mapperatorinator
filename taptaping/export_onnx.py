"""Mapperatorinator 모델을 ONNX(인코더 / 디코더)로 내보내고 int8 로 양자화한 뒤, PyTorch 와 결과를 대조한다.

Unity Inference Engine(Sentis)에서 돌리기 위한 첫 단계 (A3: 가능성 확인).
  - encoder.onnx : 멜 스펙트로그램 [1, 128, T] → 인코더 출력 [1, T/2, 512]
                   (멜 계산은 기기에서 C# 으로 — 모델 안의 torchaudio 멜은 ONNX 에 넣지 않는다)
  - decoder.onnx : 토큰 [1, L] + 인코더 출력 → 다음 토큰 로짓 [1, L, vocab]
                   (KV 캐시 없는 단순판. 매 토큰마다 앞부분을 다시 계산하므로 느리지만 내보내기·검증이 쉽다)

사용 예:
  python taptaping/export_onnx.py --model OliBomby/Mapperatorinator-v32-mini --subfolder gamemode=3 --out exports/v32mini_mania
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from osuT5.osuT5.model import Mapperatorinator


class DecomposedRMSNorm(nn.Module):
    """nn.RMSNorm 과 같은 계산을 기본 연산(Pow/ReduceMean/Sqrt/Div/Mul)으로 — 낮은 opset·Sentis 호환용"""

    def __init__(self, src: nn.RMSNorm):
        super().__init__()
        self.weight = src.weight
        self.eps = src.eps if src.eps is not None else torch.finfo(torch.float32).eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x / torch.sqrt(var + self.eps) * self.weight


def replace_rmsnorm(module: nn.Module) -> int:
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, DecomposedRMSNorm(child))
            count += 1
        else:
            count += replace_rmsnorm(child)
    return count


class EncoderWrapper(nn.Module):
    def __init__(self, model: Mapperatorinator):
        super().__init__()
        self.encoder = model.transformer.get_encoder()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # mel: [B, n_mels, T]
        return self.encoder(mel, return_dict=True).last_hidden_state


class DecoderWrapper(nn.Module):
    def __init__(self, model: Mapperatorinator):
        super().__init__()
        self.embed = model.decoder_embedder
        self.decoder = model.transformer.get_decoder()
        self.proj_out = model.transformer.proj_out

    def forward(self, input_ids: torch.Tensor, encoder_hidden: torch.Tensor) -> torch.Tensor:
        h = self.decoder(inputs_embeds=self.embed(input_ids), encoder_hidden_states=encoder_hidden,
                         use_cache=False, return_dict=True).last_hidden_state
        return self.proj_out(h)


def file_mb(p: Path) -> float:
    total = p.stat().st_size
    data = p.with_name(p.name + ".data")
    if data.exists():
        total += data.stat().st_size
    return total / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OliBomby/Mapperatorinator-v32-mini")
    ap.add_argument("--subfolder", default="gamemode=3")
    ap.add_argument("--out", default="exports/v32mini_mania")
    ap.add_argument("--mel-frames", type=int, default=2048, help="인코더 창 길이 (멜 프레임 수)")
    ap.add_argument("--dec-len", type=int, default=64, help="검증용 디코더 토큰 길이")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = Mapperatorinator.from_pretrained(args.model, subfolder=args.subfolder, attn_implementation="sdpa").eval()
    cfg = model.config
    torch.manual_seed(0)
    probe = torch.randn(1, 8, cfg.hidden_size)
    norm0 = model.transformer.get_encoder().layer_norm
    before = norm0(probe)
    n = replace_rmsnorm(model)
    after = model.transformer.get_encoder().layer_norm(probe)
    print(f"RMSNorm {n}개를 기본 연산으로 교체 (오차 {float((before - after).abs().max()):.1e})")
    print(f"파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, 입력 어휘 {cfg.vocab_size_in}, 출력 어휘 {cfg.vocab_size}")

    enc, dec = EncoderWrapper(model).eval(), DecoderWrapper(model).eval()
    torch.manual_seed(0)
    mel = torch.randn(1, cfg.n_mels, args.mel_frames)
    ids = torch.randint(0, cfg.vocab_size_in, (1, args.dec_len))
    with torch.no_grad():
        ref_h = enc(mel)
        ref_logits = dec(ids, ref_h)
    print(f"인코더 출력 {tuple(ref_h.shape)}, 디코더 출력 {tuple(ref_logits.shape)}")

    enc_path, dec_path = out / "encoder.onnx", out / "decoder.onnx"
    with torch.no_grad():
        torch.onnx.export(enc, (mel,), enc_path, input_names=["mel"], output_names=["encoder_hidden"],
                          dynamic_axes={"mel": {2: "frames"}, "encoder_hidden": {1: "enc_len"}},
                          opset_version=args.opset, dynamo=False)
        torch.onnx.export(dec, (ids, ref_h), dec_path, input_names=["input_ids", "encoder_hidden"],
                          output_names=["logits"],
                          dynamic_axes={"input_ids": {1: "dec_len"}, "encoder_hidden": {1: "enc_len"},
                                        "logits": {1: "dec_len"}},
                          opset_version=args.opset, dynamo=False)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnxruntime as ort

    results = []
    for name, path in (("encoder", enc_path), ("decoder", dec_path)):
        q_path = path.with_name(path.stem + ".int8.onnx")
        quantize_dynamic(str(path), str(q_path), weight_type=QuantType.QInt8)
        for variant, p in (("fp32", path), ("int8", q_path)):
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            feeds = {"mel": mel.numpy()} if name == "encoder" else {"input_ids": ids.numpy(), "encoder_hidden": ref_h.numpy()}
            t = time.perf_counter()
            got = sess.run(None, feeds)[0]
            ms = (time.perf_counter() - t) * 1000
            ref = (ref_h if name == "encoder" else ref_logits).numpy()
            err = float(np.max(np.abs(got - ref)))
            agree = ""
            if name == "decoder":
                agree = f", 다음 토큰 일치율 {float(np.mean(got.argmax(-1) == ref.argmax(-1))) * 100:.1f}%"
            results.append(f"  {name:<8}{variant:<5} {file_mb(p):7.1f} MB  CPU {ms:7.1f} ms  최대 오차 {err:.2e}{agree}")

    print("\n결과:")
    print("\n".join(results))


if __name__ == "__main__":
    main()
