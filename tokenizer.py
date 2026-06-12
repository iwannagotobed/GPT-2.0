"""문자 단위(char-level) 토크나이저."""

from __future__ import annotations

import pickle
from pathlib import Path


class CharTokenizer:
    """텍스트의 고유 문자로 stoi/itos 사전을 만드는 문자 단위 토크나이저."""

    def __init__(self, stoi: dict[str, int], itos: dict[int, str]) -> None:
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        """텍스트에서 정렬된 고유 문자 집합으로 토크나이저를 구축한다."""
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    def encode(self, text: str) -> list[int]:
        """문자열 → 정수 ID 리스트."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """정수 ID 리스트 → 문자열."""
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        """stoi/itos를 pickle(meta)로 저장한다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"stoi": self.stoi, "itos": self.itos}, f)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """meta.pkl에서 토크나이저를 복원한다."""
        with Path(path).open("rb") as f:
            meta = pickle.load(f)
        return cls(meta["stoi"], meta["itos"])
