from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable
from uuid import uuid4

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            # Use a unique suffix so two stores with the same classroom name do
            # not leak records into each other in Chroma's shared process state.
            client = chromadb.EphemeralClient()
            chroma_name = f"{collection_name}-{uuid4().hex[:10]}"
            self._collection = client.create_collection(
                name=chroma_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = {
            str(key): self._normalise_metadata_value(value)
            for key, value in dict(doc.metadata or {}).items()
            if value is not None
        }
        metadata.setdefault("doc_id", doc.id)
        record = {
            "id": doc.id,
            "storage_id": f"{doc.id}::{self._next_index}",
            "content": doc.content,
            "metadata": metadata,
            "embedding": [float(value) for value in self._embedding_fn(doc.content)],
        }
        self._next_index += 1
        return record

    @staticmethod
    def _normalise_metadata_value(value: Any) -> Any:
        """Convert metadata to scalar values accepted by every backend."""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not records:
            return []
        query_embedding = self._embedding_fn(query)
        ranked = sorted(
            records,
            key=lambda record: _dot(query_embedding, record["embedding"]),
            reverse=True,
        )[:top_k]
        return [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": dict(record["metadata"]),
                "score": float(_dot(query_embedding, record["embedding"])),
            }
            for record in ranked
        ]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        records = [self._make_record(doc) for doc in docs]
        if not records:
            return

        # Keep a backend-neutral copy. Besides making filtering deterministic,
        # this guarantees the public API has identical dot-product semantics
        # whether or not optional ChromaDB is installed.
        self._store.extend(records)
        if self._use_chroma and self._collection is not None:
            try:
                self._collection.add(
                    ids=[record["storage_id"] for record in records],
                    documents=[record["content"] for record in records],
                    embeddings=[record["embedding"] for record in records],
                    metadatas=[record["metadata"] for record in records],
                )
            except Exception:
                # Chroma is an optional acceleration/backend. If it rejects an
                # environment-specific value, the fully populated memory store
                # remains correct and usable.
                self._use_chroma = False
                self._collection = None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)
        normalised_filter = {
            str(key): self._normalise_metadata_value(value)
            for key, value in metadata_filter.items()
        }
        candidates = [
            record
            for record in self._store
            if all(record["metadata"].get(key) == value for key, value in normalised_filter.items())
        ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        matching = [
            record for record in self._store if record["metadata"].get("doc_id") == doc_id
        ]
        if not matching:
            return False

        matching_storage_ids = {record["storage_id"] for record in matching}
        self._store = [
            record for record in self._store if record["storage_id"] not in matching_storage_ids
        ]
        if self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(ids=sorted(matching_storage_ids))
            except Exception:
                # The memory store is the source of truth for this wrapper.
                self._use_chroma = False
                self._collection = None
        return True
