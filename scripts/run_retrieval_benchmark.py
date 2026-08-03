"""Run the locked K4 retrieval benchmark and optionally save a JSON artifact.

Examples:
    python3 scripts/run_retrieval_benchmark.py --provider local --strategy recursive
    python3 scripts/run_retrieval_benchmark.py --provider openai --strategy recursive \
        --output results/lai-recursive-700-team-v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from ingest import chunk_document, load_documents  # noqa: E402
from src import (  # noqa: E402
    FixedSizeChunker,
    LocalEmbedder,
    MarkdownHeadingChunker,
    OpenAIEmbedder,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)
from src.store import EmbeddingStore  # noqa: E402


class CachedEmbedder:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.cache: dict[str, list[float]] = {}
        self._backend_name = getattr(backend, "_backend_name", backend.__class__.__name__)

    def __call__(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = self.backend(text)
        return self.cache[text]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default="benchmarks/tiktok_shop_team_v1.json",
        help="Locked benchmark JSON, relative to repository root.",
    )
    parser.add_argument("--provider", choices=("openai", "local", "mock"), default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--strategy",
        choices=("fixed", "sentence", "recursive", "heading"),
        default="recursive",
    )
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def select_embedder(provider: str, model: str | None):
    if provider == "openai":
        return OpenAIEmbedder(model_name=model or "text-embedding-3-small")
    if provider == "local":
        return LocalEmbedder(model_name=model) if model else LocalEmbedder()
    return _mock_embed


def select_chunker(strategy: str, chunk_size: int):
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=min(100, chunk_size // 5))
    if strategy == "sentence":
        return SentenceChunker(max_sentences_per_chunk=3)
    if strategy == "heading":
        return MarkdownHeadingChunker(chunk_size=chunk_size)
    return RecursiveChunker(chunk_size=chunk_size)


def result_is_relevant(result: dict, case: dict) -> bool:
    if result["metadata"].get("doc_id") not in case["expected_doc_ids"]:
        return False
    content = result["content"].casefold()
    return all(term.casefold() in content for term in case["required_terms"])


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    benchmark_path = ROOT / args.benchmark
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    corpus = ROOT / benchmark["corpus"]

    embedder = CachedEmbedder(select_embedder(args.provider, args.model))
    chunker = select_chunker(args.strategy, args.chunk_size)
    chunks = []
    for document in load_documents(corpus):
        chunks.extend(chunk_document(document, chunker))
    store = EmbeddingStore(
        collection_name=f"team-{args.strategy}-{args.chunk_size}",
        embedding_fn=embedder,
    )
    store.add_documents(chunks)

    rows = []
    reciprocal_rank_sum = 0.0
    hits = 0
    top_k = int(benchmark.get("top_k", 3))
    for case in benchmark["queries"]:
        metadata_filter = case.get("metadata_filter")
        results = (
            store.search_with_filter(case["query"], top_k, metadata_filter)
            if metadata_filter
            else store.search(case["query"], top_k)
        )
        relevant_rank = next(
            (rank for rank, result in enumerate(results, 1) if result_is_relevant(result, case)),
            None,
        )
        if relevant_rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / relevant_rank
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "gold_answer": case["gold_answer"],
                "metadata_filter": metadata_filter,
                "expected_doc_ids": case["expected_doc_ids"],
                "required_terms": case["required_terms"],
                "hit_at_k": relevant_rank is not None,
                "first_relevant_rank": relevant_rank,
                "results": [
                    {
                        "rank": rank,
                        "doc_id": result["metadata"].get("doc_id"),
                        "chunk_index": result["metadata"].get("chunk_index"),
                        "score": round(float(result["score"]), 6),
                        "source_url": result["metadata"].get("source_url"),
                        "preview": result["content"][:250].replace("\n", " "),
                    }
                    for rank, result in enumerate(results, 1)
                ],
            }
        )

    query_count = len(rows)
    artifact = {
        "benchmark": benchmark["name"],
        "provider": args.provider,
        "backend": embedder._backend_name,
        "strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "top_k": top_k,
        "metrics": {
            "hits": hits,
            "query_count": query_count,
            "hit_rate_at_k": hits / query_count if query_count else 0.0,
            "mrr": reciprocal_rank_sum / query_count if query_count else 0.0,
        },
        "queries": rows,
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    if args.output:
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
