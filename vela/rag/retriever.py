from pathlib import Path

import chromadb
from chromadb.config import Settings

DB_PATH = str(Path(__file__).parents[2] / ".vela_db")
COLLECTION_NAME = "vela_docs"
TOP_K = 3


class Retriever:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def add_chunks(self, chunks: list[str], source: str = "") -> None:
        existing_ids = set(self._collection.get()["ids"])
        new_ids = []
        new_docs = []

        for i, chunk in enumerate(chunks):
            doc_id = f"{source}_{i}"
            if doc_id not in existing_ids:
                new_ids.append(doc_id)
                new_docs.append(chunk)

        if new_docs:
            self._collection.add(
                documents=new_docs,
                ids=new_ids,
                metadatas=[{"source": source} for _ in new_docs],
            )

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
        )
        return results["documents"][0] if results["documents"] else []

