"""미니 GPT-2 모델: CausalSelfAttention, FeedForward, Block, GPT."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    """미래 토큰을 마스킹하는 멀티헤드 self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd는 n_head로 나눠떨어져야 함"
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Q, K, V를 한 번에 projection (n_embd → 3 * n_embd)
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 마스킹 마스크 (1=참고 가능, 0=미래라 차단). buffer라 학습 대상 아님.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # (batch, time, n_embd)
        head_dim = C // self.n_head

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)  # 각 (B, T, C)
        # (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # attention 점수 (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 헤드 병합 → (B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """토큰별 비선형 변환: Linear → GELU → Linear."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer 블록 (pre-norm + residual)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # 잔차연결
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """미니 GPT-2: 토큰/위치 임베딩 → Block × N → LayerNorm → LM Head."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape  # (batch, time)
        assert T <= self.config.block_size, "시퀀스가 block_size를 초과함"

        pos = torch.arange(T, device=idx.device)  # (T,)
        x = self.token_emb(idx) + self.pos_emb(pos)  # (B, T, n_embd)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross-entropy를 위해 (B*T, vocab)와 (B*T,)로 평탄화
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """idx (B, T)에서 시작해 max_new_tokens개를 자기회귀적으로 생성한다."""
        self.eval()
        for _ in range(max_new_tokens):
            # block_size 이내로 컨텍스트 크롭
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 마지막 step만 (B, vocab)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
