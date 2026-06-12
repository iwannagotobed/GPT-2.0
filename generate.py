"""학습된 체크포인트를 로드해 텍스트를 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import get_device
from model import GPT
from tokenizer import CharTokenizer
from utils import load_checkpoint

ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "checkpoints" / "model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="미니 GPT-2 텍스트 생성")
    parser.add_argument("--prompt", type=str, default="\n", help="시작 프롬프트")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)

    ckpt = load_checkpoint(CKPT_PATH, map_location=device)
    mcfg = ckpt["model_config"]
    meta = ckpt["meta"]
    tokenizer = CharTokenizer(meta["stoi"], meta["itos"])

    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    start_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)  # (1, T)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
