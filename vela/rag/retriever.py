from pathlib import Path

import chromadb
from chromadb.config import Settings

from vela.core.embedder import Embedder

DB_PATH = str(Path(__file__).parents[2] / ".vela_db")
COLLECTION_NAME = "vela_docs_v2"  # v2: sentence-transformers 임베딩으로 통일
TOP_K = 3


class Retriever:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder or Embedder()
        self._client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        # embedding_function=None: 임베딩을 직접 계산해서 전달
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[str], source: str = "") -> None:
        existing_ids = set(self._collection.get()["ids"])
        new_ids, new_docs = [], []

        for i, chunk in enumerate(chunks):
            doc_id = f"{source}_{i}"
            if doc_id not in existing_ids:
                new_ids.append(doc_id)
                new_docs.append(chunk)

        if new_docs:
            embeddings = self._embedder.embed(new_docs).tolist()
            self._collection.add(
                documents=new_docs,
                embeddings=embeddings,
                ids=new_ids,
                metadatas=[{"source": source} for _ in new_docs],
            )

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        count = self._collection.count()
        if count == 0:
            return []

        query_emb = self._embedder.embed([query])[0].tolist()
        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, count),
            include=["documents"],
        )
        if not results["documents"]:
            return []

        return results["documents"][0]
