"""한국어 위키문헌 공개 고전소설 → 문자 단위 train.bin / val.bin 생성."""

from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

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
SOURCES_PATH = DATA_DIR / "sources.json"

API_URL = "https://ko.wikisource.org/w/api.php"
PAGE_URL = "https://ko.wikisource.org/wiki/"
LICENSE = "Public domain text hosted by Korean Wikisource; page contributions are CC BY-SA."
DEFAULT_AUTHORS = ("김동인", "현진건", "나도향", "최서해", "이상")
USER_AGENT = "mini-gpt2-korean-corpus/1.0 (educational project)"

SKIP_TAGS = {
    "audio",
    "figcaption",
    "figure",
    "nav",
    "script",
    "style",
    "table",
}
SKIP_CLASSES = {
    "catlinks",
    "licensecontainer",
    "mw-editsection",
    "mw-empty-elt",
    "noprint",
    "references",
    "sistersitebox",
}
DROP_LINE_PATTERNS = (
    re.compile(r"^저자:"),
    re.compile(r"^이 저작물은 저자가 사망한"),
    re.compile(r"^이 저작물은 미국에서"),
    re.compile(r"^퍼블릭 도메인"),
    re.compile(r"^라이선스$"),
)


class ArticleTextParser(HTMLParser):
    """MediaWiki가 렌더링한 본문 HTML에서 읽을 수 있는 텍스트만 추출한다."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").lower().split())
        if self.skip_depth or tag in SKIP_TAGS or classes & SKIP_CLASSES:
            self.skip_depth += 1
            return
        if tag in {"br", "div", "h1", "h2", "h3", "h4", "li", "p"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            self.skip_depth -= 1
            return
        if tag in {"div", "h1", "h2", "h3", "h4", "li", "p"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


class WikiSourceClient:
    """속도 제한과 일시 오류를 처리하는 한국어 위키문헌 API 클라이언트."""

    def __init__(self, request_delay: float) -> None:
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def get(self, params: dict[str, str | int], attempts: int = 6) -> dict:
        params = {"format": "json", "maxlag": 5, **params}
        for attempt in range(attempts):
            time.sleep(self.request_delay)
            try:
                response = self.session.get(API_URL, params=params, timeout=60)
            except requests.RequestException:
                if attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)
                continue
            too_many = "too many requests" in response.text.lower()
            if response.status_code not in {429, 500, 502, 503, 504} and not too_many:
                response.raise_for_status()
                payload = response.json()
                if "error" not in payload:
                    return payload
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"위키문헌 API 오류 ({response.status_code}): {response.text[:300]}"
                )
            retry_after = int(response.headers.get("Retry-After", 0))
            time.sleep(max(retry_after, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")

    def author_novels(self, author: str) -> list[dict]:
        """저자 문서를 참조하며 소설 분류가 붙은 본문 페이지를 반환한다."""
        backlinks: list[dict] = []
        continuation: str | None = None
        while True:
            params: dict[str, str | int] = {
                "action": "query",
                "list": "backlinks",
                "bltitle": f"저자:{author}",
                "blnamespace": 0,
                "bllimit": 500,
            }
            if continuation:
                params["blcontinue"] = continuation
            payload = self.get(params)
            backlinks.extend(payload["query"]["backlinks"])
            continuation = payload.get("continue", {}).get("blcontinue")
            if not continuation:
                break

        novels: list[dict] = []
        for start in range(0, len(backlinks), 50):
            batch = backlinks[start : start + 50]
            payload = self.get(
                {
                    "action": "query",
                    "pageids": "|".join(str(item["pageid"]) for item in batch),
                    "prop": "info|categories|revisions",
                    "cllimit": 500,
                    "rvprop": "ids",
                }
            )
            for page in payload["query"]["pages"].values():
                categories = [item["title"].removeprefix("분류:") for item in page.get("categories", [])]
                if not any(category.endswith("소설") for category in categories):
                    continue
                novels.append(
                    {
                        "author": author,
                        "pageid": int(page["pageid"]),
                        "title": page["title"],
                        "size": int(page["length"]),
                        "revision_id": int(page["revisions"][0]["revid"]),
                    }
                )
        return novels

    def parse_revision(self, revision_id: int) -> str:
        payload = self.get(
            {
                "action": "parse",
                "oldid": revision_id,
                "prop": "text",
                "disableeditsection": 1,
                "disabletoc": 1,
            }
        )
        parser = ArticleTextParser()
        parser.feed(payload["parse"]["text"]["*"])
        return clean_text(parser.text())


def clean_text(raw: str) -> str:
    """렌더링 잔여물과 과도한 공백을 제거하고 NFC로 정규화한다."""
    text = unicodedata.normalize("NFC", html.unescape(raw)).replace("\xa0", " ")
    lines: list[str] = []
    previous_blank = True
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if any(pattern.search(line) for pattern in DROP_LINE_PATTERNS):
            continue
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    return "\n".join(lines).strip()


def discover_sources(
    client: WikiSourceClient,
    authors: tuple[str, ...],
    target_chars: int,
    min_document_chars: int,
) -> list[dict]:
    """공개 작가별 소설 검색 결과에서 본문 문서를 고정 revision으로 수집한다."""
    candidates: dict[int, dict] = {}
    for author in authors:
        print(f"검색 중: {author}")
        for result in client.author_novels(author):
            if int(result["size"]) < min_document_chars:
                continue
            candidates[int(result["pageid"])] = result

    selected: list[dict] = []
    total = 0
    for candidate in sorted(candidates.values(), key=lambda item: item["title"]):
        revision_id = int(candidate["revision_id"])
        text = client.parse_revision(revision_id)
        if len(text) < min_document_chars:
            continue
        selected.append(
            {
                "author": candidate["author"],
                "title": candidate["title"],
                "pageid": int(candidate["pageid"]),
                "revision_id": revision_id,
                "url": PAGE_URL + quote(candidate["title"].replace(" ", "_")),
                "characters": len(text),
                "license": LICENSE,
                "text": text,
            }
        )
        total += len(text)
        print(f"  {candidate['title']}: {len(text):,}자 (누계 {total:,}자)")
        if total >= target_chars and len(selected) >= 2:
            return selected
    raise RuntimeError(f"목표 {target_chars:,}자를 확보하지 못했습니다. 확보: {total:,}자")


def load_pinned_sources(client: WikiSourceClient) -> list[dict]:
    """sources.json에 기록된 revision을 다시 읽어 동일한 코퍼스를 재현한다."""
    manifest = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    sources: list[dict] = []
    for item in manifest["sources"]:
        text = client.parse_revision(int(item["revision_id"]))
        source = {**item, "characters": len(text), "text": text}
        sources.append(source)
        print(f"재현 중: {item['title']} ({len(text):,}자)")
    return sources


def assign_splits(sources: list[dict], val_ratio: float = 0.1) -> None:
    """작품 단위로 뒤쪽 약 10%를 validation에 배정한다."""
    val_target = sum(source["characters"] for source in sources) * val_ratio
    val_chars = 0
    for index, source in reversed(list(enumerate(sources))):
        source["split"] = "val" if index > 0 and val_chars < val_target else "train"
        if source["split"] == "val":
            val_chars += source["characters"]


def document_text(source: dict) -> str:
    return f"\n\n===== {source['title']} / {source['author']} =====\n\n{source['text']}\n"


def write_manifest(sources: list[dict], target_chars: int) -> None:
    manifest_sources = [{key: value for key, value in source.items() if key != "text"} for source in sources]
    manifest = {
        "dataset": "Korean public-domain novels from Korean Wikisource",
        "api": API_URL,
        "target_characters": target_chars,
        "total_characters": sum(source["characters"] for source in sources),
        "sources": manifest_sources,
    }
    SOURCES_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare(sources: list[dict], target_chars: int) -> None:
    assign_splits(sources)
    train_text = "".join(document_text(source) for source in sources if source["split"] == "train")
    val_text = "".join(document_text(source) for source in sources if source["split"] == "val")
    text = train_text + val_text
    if len(text) < target_chars:
        raise ValueError(f"정제 후 코퍼스가 목표보다 작습니다: {len(text):,} < {target_chars:,}")

    INPUT_PATH.write_text(text, encoding="utf-8")
    tokenizer = CharTokenizer.build(text)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(f"vocab {tokenizer.vocab_size:,}개는 uint16 범위를 초과합니다")
    sample = text[:1000]
    assert tokenizer.decode(tokenizer.encode(sample)) == sample, "라운드트립 불일치"

    train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint16)
    val_ids = np.array(tokenizer.encode(val_text), dtype=np.uint16)
    if len(train_ids) <= 128 or len(val_ids) <= 128:
        raise ValueError("train/val 데이터가 block_size=128보다 길어야 합니다")
    train_ids.tofile(TRAIN_PATH)
    val_ids.tofile(VAL_PATH)
    tokenizer.save(META_PATH)
    write_manifest(sources, target_chars)

    print(f"전체 문자 수: {len(text):,}")
    print(f"작품 수: {len(sources)} (train {sum(s['split'] == 'train' for s in sources)}, val {sum(s['split'] == 'val' for s in sources)})")
    print(f"vocab 크기: {tokenizer.vocab_size:,}")
    print(f"train.bin: {len(train_ids):,} 토큰")
    print(f"val.bin:   {len(val_ids):,} 토큰")


def main() -> None:
    parser = argparse.ArgumentParser(description="한국어 공개 고전소설 코퍼스 준비")
    parser.add_argument("--target-chars", type=int, default=1_000_000)
    parser.add_argument("--min-document-chars", type=int, default=4_000)
    parser.add_argument("--request-delay", type=float, default=2.0)
    parser.add_argument("--refresh", action="store_true", help="고정 manifest를 무시하고 작품을 다시 검색")
    args = parser.parse_args()

    client = WikiSourceClient(args.request_delay)
    if SOURCES_PATH.exists() and not args.refresh:
        sources = load_pinned_sources(client)
    else:
        sources = discover_sources(
            client,
            DEFAULT_AUTHORS,
            args.target_chars,
            args.min_document_chars,
        )
    prepare(sources, args.target_chars)


if __name__ == "__main__":
    main()
