"""미니 GPT-2 하이퍼파라미터 및 device 선택."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GPTConfig:
    """모델 구조 하이퍼파라미터. vocab_size는 런타임에 meta.pkl에서 주입한다."""

    vocab_size: int = 0  # prepare.py가 만든 vocab 크기로 덮어씀
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    block_size: int = 128
    dropout: float = 0.0  # 필수 단계는 0, 선택 단계에서 0.1 등으로 상향


@dataclass
class TrainConfig:
    """학습 루프 하이퍼파라미터."""

    batch_size: int = 32
    max_iters: int = 5000
    lr: float = 3e-4
    eval_interval: int = 500
    eval_iters: int = 200  # estimate_loss 평균에 사용할 배치 수
    seed: int = 1337


def get_device() -> str:
    """cuda > mps > cpu 순서로 사용 가능한 device를 반환한다."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
