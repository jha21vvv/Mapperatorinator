"""extract.py 결과(npz) → 학습 샘플 (4박 멜 + 토큰 시퀀스)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from model import (ACT0, BLOCK_BEATS, BOS, CTX_BEATS, EOS, HOLD0, MAX_CTX_TOKENS, MAX_OUT_TOKENS, PAD, SEP, SUB, TIME0,
                   WIN, action_token)


def block_tokens(notes_by_pos, start_pos, n_pos):
    """[start_pos, start_pos+n_pos) 칸의 노트 → [시간, 행동, 시간, 행동, ...]"""
    out = []
    for p in range(start_pos, start_pos + n_pos):
        st = notes_by_pos.get(p)
        if st:
            out += [TIME0 + p - start_pos, action_token(st)]
    return out


def held_state(holds, pos):
    """pos 시점(블록 시작)에 누르고 있는 키 (롱노트 시작 < pos <= 끝 직전)"""
    v = 0
    for col, s, e in holds:
        if s < pos <= e:  # 블록 앞에서 시작했고, 끝(롱끝 이벤트)이 이 블록 시작 이후 → 들어올 때 누르고 있음
            v |= 1 << col
    return HOLD0 + v


class Song:
    def __init__(self, path: Path):
        z = np.load(path)
        self.mel = z["mel"]  # [nb*48, 80] float16
        self.meta = json.loads(str(z["meta"]))
        self.charts = []
        for c in json.loads(str(z["charts"])):
            by_pos, holds, open_ = {}, [], {}
            for pos, col, kind in c["notes"]:
                st = by_pos.setdefault(pos, [0, 0, 0, 0])
                st[col] = kind
                if kind == 2:
                    open_[col] = pos
                elif kind == 3 and col in open_:
                    holds.append((col, open_.pop(col), pos))
            self.charts.append({"star": float(c["star"]), "by_pos": by_pos, "holds": holds, "n": len(c["notes"])})
        self.n_beats = self.mel.shape[0] // SUB


def load_songs(folder: Path, split: str, limit=None):
    songs = []
    for p in sorted((folder / "songs").glob("*.npz")):
        meta = json.loads(str(np.load(p)["meta"]))
        if (split == "eval") != bool(meta.get("eval")):
            continue
        songs.append(Song(p))
        if limit and len(songs) >= limit:
            break
    return songs


def make_sample(song: Song, chart: dict, k: int):
    """박 k 부터 만들 2박 (앞 2박은 소리·노트 문맥)"""
    s0 = (k - CTX_BEATS) * SUB  # 인코더 창 시작 칸
    mel = np.full((WIN, song.mel.shape[1]), np.log(1e-6), np.float32)
    a, b = max(s0, 0), min(s0 + WIN, song.mel.shape[0])
    if b > a:
        mel[a - s0: b - s0] = song.mel[a:b]
    gen_pos = k * SUB
    ctx = block_tokens(chart["by_pos"], gen_pos - CTX_BEATS * SUB, CTX_BEATS * SUB)[-MAX_CTX_TOKENS:]
    tgt = block_tokens(chart["by_pos"], gen_pos, BLOCK_BEATS * SUB)[:MAX_OUT_TOKENS]
    seq = [BOS] + ctx + [SEP, held_state(chart["holds"], gen_pos)] + tgt + [EOS]
    n_prefix = 1 + len(ctx) + 2
    return mel, seq, n_prefix


class Batcher:
    """곡·악보·시작 박을 무작위로 골라 배치를 만든다 (빈 구간만 계속 나오지 않게 노트 있는 블록 위주)."""

    def __init__(self, songs, batch_size=32, seed=0, empty_keep=0.15):
        self.songs, self.bs = songs, batch_size
        self.rng = random.Random(seed)
        self.empty_keep = empty_keep
        self.pairs = [(s, c) for s in songs for c in s.charts]
        self.weights = [c["n"] ** 0.5 for _s, c in self.pairs]

    def sample(self):
        while True:
            s, c = self.rng.choices(self.pairs, self.weights)[0]
            k = self.rng.randrange(0, max(1, s.n_beats - BLOCK_BEATS))
            mel, seq, npre = make_sample(s, c, k)
            if len(seq) - npre <= 1 and self.rng.random() > self.empty_keep:
                continue
            return mel, seq, npre, c["star"]

    def batch(self):
        items = [self.sample() for _ in range(self.bs)]
        L = max(len(x[1]) for x in items)
        mel = torch.from_numpy(np.stack([x[0] for x in items]))
        tok = torch.full((self.bs, L), PAD, dtype=torch.long)
        loss_mask = torch.zeros((self.bs, L), dtype=torch.bool)
        for i, (_m, seq, npre, _s) in enumerate(items):
            tok[i, : len(seq)] = torch.tensor(seq)
            loss_mask[i, npre - 1: len(seq) - 1] = True  # 다음 토큰 예측: 위치 t 가 토큰 t+1 을 맞힘 (만들 부분만)
        star = torch.tensor([x[3] for x in items], dtype=torch.float32)
        return mel, tok, loss_mask, star
