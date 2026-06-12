"""미니 GPT-2 학습 루프: AdamW + train/val loss 로깅 + 체크포인트 저장."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import torch

from config import GPTConfig, TrainConfig, get_device
from model import GPT
from utils import estimate_loss, get_batch, load_split, perplexity, save_checkpoint

ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "checkpoints" / "model.pt"
META_PATH = ROOT / "data" / "meta.pkl"
LOG_PATH = ROOT / "loss_log.csv"


def main() -> None:
    device = get_device()
    print(f"device: {device}")

    tcfg = TrainConfig()
    torch.manual_seed(tcfg.seed)

    # vocab 로드 → 모델 config 주입
    with META_PATH.open("rb") as f:
        meta = pickle.load(f)
    vocab_size = len(meta["stoi"])
    mcfg = GPTConfig(vocab_size=vocab_size)
    print(f"vocab_size: {vocab_size}")

    train_data = load_split("train")
    val_data = load_split("val")
    splits = {"train": train_data, "val": val_data}

    model = GPT(mcfg).to(device)
    print(f"파라미터 수: {model.num_params() / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)

    best_val = float("inf")
    log_rows: list[dict] = []

    for it in range(tcfg.max_iters + 1):
        # 주기적 평가 + 로깅 + 체크포인트
        if it % tcfg.eval_interval == 0 or it == tcfg.max_iters:
            losses = estimate_loss(
                model, splits, mcfg.block_size, tcfg.batch_size, tcfg.eval_iters, device
            )
            ppl = perplexity(losses["val"])
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | val ppl {ppl:.2f}"
            )
            log_rows.append({"iter": it, "train": losses["train"], "val": losses["val"], "val_ppl": ppl})
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(CKPT_PATH, model, mcfg, meta)

        if it == tcfg.max_iters:
            break

        # 학습 스텝
        x, y = get_batch(train_data, mcfg.block_size, tcfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # loss 곡선 재현용 CSV 저장
    with LOG_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "train", "val", "val_ppl"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"학습 완료. best val loss {best_val:.4f} → {CKPT_PATH}")
    print(f"loss 로그 저장: {LOG_PATH}")


if __name__ == "__main__":
    main()
