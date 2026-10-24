"""Build the vector index from scraped listings in `data/raw/`.

Supported inputs: `.json` (a list of objects, or `{"listings": [...]}`),
`.jsonl`, and `.csv`. Run with `uv run pi-ingest`.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterator

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from property_intelligence_rag import config
from property_intelligence_rag.schema import Property


def get_embeddings() -> Embeddings:
    # "fake" lets tests run without network access.
    if config.EMBEDDING_MODEL == "fake":
        return DeterministicFakeEmbedding(size=256)
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        # OpenRouter model ids aren't tiktoken names; send raw text instead of token ids.
        check_embedding_ctx_length=False,
    )


def iter_raw_records(raw_dir: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    for path in sorted(raw_dir.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                data = data.get("listings") or data.get("results") or [data]
            records = data
        elif suffix == ".jsonl":
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        elif suffix == ".csv":
            with path.open(newline="") as f:
                records = list(csv.DictReader(f))
        else:
            continue
        for i, record in enumerate(records):
            yield f"{path.stem}-{i}", record


def load_documents(raw_dir: Path = config.RAW_DIR) -> list[Document]:
    docs: dict[str, Document] = {}
    for fallback_id, record in iter_raw_records(raw_dir):
        prop = Property.from_raw(record, fallback_id)
        docs[prop.id] = prop.to_document()  # later scrapes of the same id win
    return list(docs.values())


def build_index(raw_dir: Path = config.RAW_DIR, index_path: Path = config.INDEX_PATH) -> int:
    docs = load_documents(raw_dir)
    if not docs:
        raise SystemExit(f"No listings found in {raw_dir}")
    store = InMemoryVectorStore(embedding=get_embeddings())
    store.add_documents(docs, ids=[d.metadata["id"] for d in docs])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(index_path))
    return len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--index", type=Path, default=config.INDEX_PATH)
    args = parser.parse_args()
    count = build_index(args.raw_dir, args.index)
    print(f"Indexed {count} listings -> {args.index}")
