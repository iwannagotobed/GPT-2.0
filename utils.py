"""배치 생성, loss 추정, perplexity, 체크포인트 I/O."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_split(split: str) -> np.ndarray:
    """train/val .bin을 memmap으로 로드한다."""
    path = DATA_DIR / f"{split}.bin"
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """무작위 위치에서 block_size 길이의 x와 한 칸 민 y를 뽑는다. 반환 shape (B, T)."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    if device == "cuda":
        # non_blocking 전송 (pin_memory)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    splits: dict[str, np.ndarray],
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    """train/val 각각 eval_iters개 배치 평균 loss를 계산한다."""
    model.eval()
    out: dict[str, float] = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def perplexity(loss: float) -> float:
    """평균 cross-entropy loss → perplexity."""
    return math.exp(loss)


def save_checkpoint(path: str | Path, model: torch.nn.Module, model_config, meta: dict) -> None:
    """모델 state, config, vocab(meta)를 함께 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """체크포인트 dict를 로드한다."""
    return torch.load(path, map_location=map_location, weights_only=False)
