"""폰(Unity Inference Engine)용 ONNX 내보내기 — KV 캐시 디코더.

만드는 파일 (모델 하나당):
  encoder.onnx  : mel [1, 128, 2048] → cross_k_0..5, cross_v_0..5  [1, 8, 1024, 64]
                  (인코더 출력에서 디코더 교차 어텐션용 K/V 까지 미리 계산 — 창마다 한 번)
  decoder.onnx  : input_ids [1, T], past_k_0..5 / past_v_0..5 [1, 8, P, 64], cross_k/v
                  → logits [1, vocab] (마지막 위치만), present_k/v [1, 8, P+T, 64]
                  T>1 이면 프롬프트 한꺼번에(prefill), T=1 이면 토큰 하나씩.
  meta.json     : 층 수, 헤드 수, 어휘 크기 등

원본 VarWhisper 디코더와 같은 계산을 내보내기 쉬운 기본 연산으로 다시 짰다 (RMSNorm·RoPE·어텐션·GELU).
PyTorch 원본과 로짓을 대조해 검증한다.

  PYTHONPATH=. python taptaping/export_kv.py --subfolder gamemode=3 --out exports/map
  PYTHONPATH=. python taptaping/export_kv.py --subfolder ""         --out exports/timing
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from osuT5.osuT5.model import Mapperatorinator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import replace_rmsnorm  # noqa: E402


def rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class EncoderKV(nn.Module):
    def __init__(self, model: Mapperatorinator):
        super().__init__()
        self.encoder = model.transformer.get_encoder()
        dec = model.transformer.get_decoder()
        self.cross = nn.ModuleList([layer.cross_attn for layer in dec.layers])
        self.h = dec.layers[0].cross_attn.num_heads
        self.d = dec.layers[0].cross_attn.head_dim

    def forward(self, mel):
        enc = self.encoder(mel, return_dict=True).last_hidden_state  # [1, S, 512]
        outs = []
        for ca in self.cross:
            kv = ca.Wkv(enc).view(1, -1, 2, self.h, self.d)
            k, v = kv.transpose(1, 3).unbind(dim=2)  # [1, H, S, D]
            outs += [k, v]
        return tuple(outs)


class DecoderKV(nn.Module):
    def __init__(self, model: Mapperatorinator):
        super().__init__()
        dec = model.transformer.get_decoder()
        self.embed = model.decoder_embedder
        self.layers = dec.layers
        self.norm = dec.layer_norm
        self.proj_out = model.transformer.proj_out
        a0 = dec.layers[0].self_attn
        self.h, self.d = a0.num_heads, a0.head_dim
        self.register_buffer("inv_freq", a0.rotary_emb.inv_freq.detach().clone().float(), persistent=False)
        self.rope_scale = float(getattr(a0.rotary_emb, "attention_scaling", 1.0))
        self.n = len(dec.layers)

    @staticmethod
    def _eps(norm):
        return norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps

    def forward(self, input_ids, *kv):
        past = kv[: 2 * self.n]
        cross = kv[2 * self.n:]
        T = input_ids.shape[1]
        P = past[0].shape[2]
        x = self.embed(input_ids)  # [1, T, 512]

        pos = torch.arange(T, dtype=torch.float32).unsqueeze(0) + P  # [1, T]
        freqs = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # [1, T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.rope_scale).unsqueeze(1)  # [1, 1, T, D]
        sin = (emb.sin() * self.rope_scale).unsqueeze(1)

        # 인과 마스크: 새 토큰 i 는 과거 전체 + 새 토큰 0..i 를 본다
        qi = torch.arange(T).view(T, 1) + P
        kj = torch.arange(P + T).view(1, P + T)
        mask = torch.where(kj <= qi, 0.0, float("-inf")).view(1, 1, T, P + T)
        scale = self.d ** -0.5

        presents = []
        for li, layer in enumerate(self.layers):
            sa = layer.self_attn
            h = rms(x, layer.self_attn_layer_norm.weight, self._eps(layer.self_attn_layer_norm))
            qkv = sa.Wqkv(h).view(1, T, 3, self.h, self.d)
            q, k, v = qkv.transpose(1, 3).unbind(dim=2)  # [1, H, T, D]
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k_all = torch.cat([past[2 * li], k], dim=2)
            v_all = torch.cat([past[2 * li + 1], v], dim=2)
            presents += [k_all, v_all]
            att = torch.softmax(q @ k_all.transpose(2, 3) * scale + mask, dim=-1) @ v_all
            x = x + sa.Wo(att.transpose(1, 2).reshape(1, T, self.h * self.d))

            ca = layer.cross_attn
            h = rms(x, layer.cross_attn_layer_norm.weight, self._eps(layer.cross_attn_layer_norm))
            q = ca.Wq(h).view(1, T, self.h, self.d).transpose(1, 2)
            att = torch.softmax(q @ cross[2 * li].transpose(2, 3) * scale, dim=-1) @ cross[2 * li + 1]
            x = x + ca.Wo(att.transpose(1, 2).reshape(1, T, self.h * self.d))

            h = rms(x, layer.final_layer_norm.weight, self._eps(layer.final_layer_norm))
            x = x + layer.fc2(torch.nn.functional.gelu(layer.fc1(h)))

        x = rms(x[:, -1:, :], self.norm.weight, self._eps(self.norm))
        logits = self.proj_out(x)[:, 0, :]  # [1, V]
        return (logits, *presents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OliBomby/Mapperatorinator-v32-mini")
    ap.add_argument("--subfolder", default="gamemode=3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    kw = {"subfolder": a.subfolder} if a.subfolder else {}
    model = Mapperatorinator.from_pretrained(a.model, attn_implementation="sdpa", **kw).eval()
    replace_rmsnorm(model.transformer.get_encoder())  # 인코더는 원본 모듈을 그대로 쓰므로 RMSNorm 만 풀어 준다
    cfg = model.config
    enc_m, dec_m = EncoderKV(model).eval(), DecoderKV(model).eval()
    H, D, n = dec_m.h, dec_m.d, dec_m.n

    torch.manual_seed(0)
    mel = torch.randn(1, cfg.n_mels, 2048)
    prompt = torch.randint(0, cfg.vocab_size_in, (1, 40))
    steps = torch.randint(0, cfg.vocab_size, (1, 6))

    # ── PyTorch 원본(캐시 없이 전체 계산)과 대조 ─────────────────────────
    with torch.no_grad():
        enc_h = model.transformer.get_encoder()(mel, return_dict=True).last_hidden_state
        full_ids = torch.cat([prompt, steps], dim=1)
        ref = model.transformer.proj_out(model.transformer.get_decoder()(
            inputs_embeds=model.decoder_embedder(full_ids), encoder_hidden_states=enc_h,
            use_cache=False, return_dict=True).last_hidden_state)[0]  # [L, V]

        cross = enc_m(mel)
        empty = [torch.zeros(1, H, 0, D) for _ in range(2 * n)]
        res = dec_m(prompt, *empty, *cross)
        errs = [float((res[0][0] - ref[prompt.shape[1] - 1]).abs().max())]
        past = list(res[1:])
        for i in range(steps.shape[1]):
            res = dec_m(steps[:, i:i + 1], *past, *cross)
            past = list(res[1:])
            errs.append(float((res[0][0] - ref[prompt.shape[1] + i]).abs().max()))
    print(f"PyTorch 재구현 vs 원본 로짓 최대 오차: {max(errs):.2e}")

    # ── ONNX 내보내기 ─────────────────────────────────────────────────
    kv_names = [f"{t}_{i}" for i in range(n) for t in ("k", "v")]
    with torch.no_grad():
        torch.onnx.export(enc_m, (mel,), out / "encoder.onnx", input_names=["mel"],
                          output_names=[f"cross_{s}" for s in kv_names],
                          opset_version=a.opset, dynamo=False)
        dyn = {"input_ids": {1: "T"}}
        for s in kv_names:
            dyn[f"past_{s}"] = {2: "P"}
            dyn[f"present_{s}"] = {2: "PT"}
        torch.onnx.export(dec_m, (prompt, *empty, *cross), out / "decoder.onnx",
                          input_names=["input_ids"] + [f"past_{s}" for s in kv_names] + [f"cross_{s}" for s in kv_names],
                          output_names=["logits"] + [f"present_{s}" for s in kv_names],
                          dynamic_axes=dyn, opset_version=a.opset, dynamo=False)

    # ── ONNX Runtime 으로 다시 대조 + 속도 ──────────────────────────────
    import onnxruntime as ort
    so = ort.SessionOptions()
    enc_s = ort.InferenceSession(str(out / "encoder.onnx"), so, providers=["CPUExecutionProvider"])
    dec_s = ort.InferenceSession(str(out / "decoder.onnx"), so, providers=["CPUExecutionProvider"])
    t = time.perf_counter()
    cross_np = enc_s.run(None, {"mel": mel.numpy()})
    enc_ms = (time.perf_counter() - t) * 1000
    feeds = {"input_ids": prompt.numpy()}
    feeds.update({f"past_{s}": np.zeros((1, H, 0, D), np.float32) for s in kv_names})
    feeds.update({f"cross_{s}": c for s, c in zip(kv_names, cross_np)})
    t = time.perf_counter()
    r = dec_s.run(None, feeds)
    prefill_ms = (time.perf_counter() - t) * 1000
    onnx_errs = [float(np.abs(r[0][0] - ref[prompt.shape[1] - 1].numpy()).max())]
    step_ms = []
    for i in range(steps.shape[1]):
        feeds["input_ids"] = steps[:, i:i + 1].numpy()
        feeds.update({f"past_{s}": p for s, p in zip(kv_names, r[1:])})
        t = time.perf_counter()
        r = dec_s.run(None, feeds)
        step_ms.append((time.perf_counter() - t) * 1000)
        onnx_errs.append(float(np.abs(r[0][0] - ref[prompt.shape[1] + i].numpy()).max()))
    print(f"ONNX vs 원본 로짓 최대 오차: {max(onnx_errs):.2e}")
    mb = lambda p: p.stat().st_size / 1e6  # noqa: E731
    print(f"encoder.onnx {mb(out / 'encoder.onnx'):.1f}MB  {enc_ms:.0f}ms/창 | decoder.onnx {mb(out / 'decoder.onnx'):.1f}MB  "
          f"prefill(40) {prefill_ms:.0f}ms, 토큰당 {np.median(step_ms):.1f}ms (CPU, ONNX Runtime)")

    meta = {"layers": n, "heads": H, "head_dim": D, "d_model": cfg.hidden_size, "vocab_out": cfg.vocab_size,
            "vocab_in": cfg.vocab_size_in, "mel_frames": 2048, "n_mels": cfg.n_mels, "enc_len": int(cross[0].shape[2]),
            "sample_rate": cfg.sample_rate, "n_fft": cfg.n_fft, "hop_length": cfg.hop_length, "f_min": cfg.f_min,
            "f_max": cfg.f_max, "source": f"{a.model}/{a.subfolder}", "max_logit_err": max(onnx_errs)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
