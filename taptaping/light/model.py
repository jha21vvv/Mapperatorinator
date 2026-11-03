"""가벼운 4K 채보 모델 (GOCT 방식 — "Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts", MIT 코드 참고해 새로 작성).

한 번에 2박씩 만든다:
  인코더 입력 : 박 정렬 로그 멜 4박 = [192 칸 × 80]  (앞 2박 + 만들 2박)
  디코더 입력 : [난이도] [앞 2박 노트 토큰…] [SEP] [지금 누르고 있는 롱노트 키 상태] → 만들 2박 노트 토큰 … [EOS]
  노트 토큰   : (시간 토큰 0~95 = 2박 안의 1/48 박 위치) + (행동 토큰 = 키 4개 각각 없음/단타/롱시작/롱끝 조합)
"""
from __future__ import annotations

import math

import torch
from torch import nn

SUB = 48            # 한 박 칸 수
BLOCK_BEATS = 2     # 만들 박 수
CTX_BEATS = 2       # 소리로 함께 보는 앞 박 수
WIN = (BLOCK_BEATS + CTX_BEATS) * SUB  # 인코더 칸 수 = 192
N_MELS = 80

# 토큰 표
PAD, BOS, EOS, SEP = 0, 1, 2, 3
TIME0 = 4                       # 4..99   : 2박 안 위치 0~95
N_TIME = BLOCK_BEATS * SUB
ACT0 = TIME0 + N_TIME           # 100..354: 행동 1~255 (키 4개 × 상태 4가지, 0=아무것도 없음 제외)
N_ACT = 255
HOLD0 = ACT0 + N_ACT            # 355..370: 누르고 있는 키 상태 0~15
VOCAB = HOLD0 + 16
MAX_CTX_TOKENS = 48             # 앞 2박 토큰 최대 (넘으면 뒤쪽만)
MAX_OUT_TOKENS = 96             # 만들 2박 토큰 최대


def action_token(states):  # states: 키 4개 상태 [0 없음, 1 단타, 2 롱시작, 3 롱끝]
    v = states[0] + 4 * states[1] + 16 * states[2] + 64 * states[3]
    return ACT0 + v - 1


def action_states(tok):
    v = tok - ACT0 + 1
    return [v % 4, (v // 4) % 4, (v // 16) % 4, (v // 64) % 4]


class LightChartModel(nn.Module):
    def __init__(self, d=192, enc_layers=3, dec_layers=3, heads=4, ff=768, dropout=0.1):
        super().__init__()
        self.d = d
        self.mel_in = nn.Linear(N_MELS, d)
        self.enc_pos = nn.Parameter(torch.zeros(1, WIN, d))
        self.tok = nn.Embedding(VOCAB, d)
        self.dec_pos = nn.Parameter(torch.zeros(1, 2 + MAX_CTX_TOKENS + 2 + MAX_OUT_TOKENS + 2, d))
        self.diff = nn.Sequential(nn.Linear(16, d), nn.GELU(), nn.Linear(d, d))  # 난이도(★) → 벡터
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True), enc_layers)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True), dec_layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, VOCAB)
        nn.init.normal_(self.enc_pos, std=0.02)
        nn.init.normal_(self.dec_pos, std=0.02)
        self.register_buffer("mel_mean", torch.zeros(N_MELS))
        self.register_buffer("mel_std", torch.ones(N_MELS))

    @staticmethod
    def star_features(star):  # [B] → [B,16] 사인 기반 표현 (0~10★)
        k = torch.arange(8, device=star.device, dtype=torch.float32)
        x = star[:, None] / 10.0 * (2 ** k)[None] * math.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

    def encode(self, mel):  # mel [B, 192, 80] (log)
        x = (mel - self.mel_mean) / self.mel_std
        return self.enc(self.mel_in(x) + self.enc_pos[:, : x.shape[1]])

    def decode(self, tokens, memory, star):  # tokens [B, L] → logits [B, L, V]
        L = tokens.shape[1]
        x = self.tok(tokens) + self.dec_pos[:, :L]
        x[:, 0] = x[:, 0] + self.diff(self.star_features(star))  # 첫 칸(BOS)에 난이도
        mask = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), 1)
        pad = tokens == PAD
        h = self.dec(x, memory, tgt_mask=mask, tgt_key_padding_mask=pad)
        return self.out(self.norm(h))

    def forward(self, mel, tokens, star):
        return self.decode(tokens, self.encode(mel), star)
