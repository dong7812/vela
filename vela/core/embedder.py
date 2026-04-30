import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = SentenceTransformer(self._MODEL_NAME)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # vectors are already L2-normalized so dot product == cosine similarity
        return float(np.dot(a, b))

    def pairwise_similarities(self, texts: list[str]) -> list[float]:
        if len(texts) < 2:
            return []
        embeddings = self.embed(texts)
        return [
            self.cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
