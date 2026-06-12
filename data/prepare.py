"""Tiny Shakespeare 다운로드 → 문자 단위 인코딩 → train.bin / val.bin / meta.pkl 생성."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import requests

# 프로젝트 루트를 import 경로에 추가 (tokenizer.py 사용)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer import CharTokenizer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
INPUT_PATH = DATA_DIR / "input.txt"
TRAIN_PATH = DATA_DIR / "train.bin"
VAL_PATH = DATA_DIR / "val.bin"
META_PATH = DATA_DIR / "meta.pkl"

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def download() -> str:
    """input.txt가 없으면 다운로드하고 전체 텍스트를 반환한다."""
    if not INPUT_PATH.exists():
        print(f"다운로드 중: {DATA_URL}")
        resp = requests.get(DATA_URL, timeout=30)
        resp.raise_for_status()
        INPUT_PATH.write_text(resp.text, encoding="utf-8")
    return INPUT_PATH.read_text(encoding="utf-8")


def main() -> None:
    text = download()
    print(f"전체 문자 수: {len(text):,}")

    tokenizer = CharTokenizer.build(text)
    print(f"vocab 크기: {tokenizer.vocab_size}")

    # encode→decode 라운드트립 검증
    sample = text[:1000]
    assert tokenizer.decode(tokenizer.encode(sample)) == sample, "라운드트립 불일치"

    ids = tokenizer.encode(text)
    n = len(ids)
    split = int(n * 0.9)
    train_ids = np.array(ids[:split], dtype=np.uint16)
    val_ids = np.array(ids[split:], dtype=np.uint16)

    train_ids.tofile(TRAIN_PATH)
    val_ids.tofile(VAL_PATH)
    tokenizer.save(META_PATH)

    print(f"train.bin: {len(train_ids):,} 토큰")
    print(f"val.bin:   {len(val_ids):,} 토큰")
    print(f"저장 완료: {TRAIN_PATH.name}, {VAL_PATH.name}, {META_PATH.name}")


if __name__ == "__main__":
    main()
