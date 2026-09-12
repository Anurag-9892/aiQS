"""
In-memory vector store powered by NumPy cosine similarity.
Supports metadata filtering, top-k ranking, and document stats.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from document_loader import DocumentChunk


class VectorStore:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Add document chunks and their corresponding embedding vectors to the store."""
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks and {len(embeddings)} embeddings")

        new_embeds = np.array(embeddings, dtype=np.float32)
        # Normalize embeddings for fast cosine similarity via dot product
        norms = np.linalg.norm(new_embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_embeds = new_embeds / norms

        if self.embeddings.size == 0:
            self.embeddings = norm_embeds
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack([self.embeddings, norm_embeds])
            self.chunks.extend(chunks)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        min_score: float = 0.0
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for the top-k most semantically similar chunks to the query vector.
        Returns a list of (DocumentChunk, similarity_score) sorted descending by score.
        """
        if self.embeddings.size == 0 or not self.chunks:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm

        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.embeddings, q_vec)

        # Get top-k indices
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_score:
                results.append((self.chunks[idx], score))

        return results

    def clear(self) -> None:
        """Clear all stored vectors and chunks."""
        self.chunks = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """Return store summary stats."""
        sources = list(set(c.source_file for c in self.chunks))
        return {
            "total_chunks": len(self.chunks),
            "total_sources": len(sources),
            "sources": sources,
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings.size > 0 else 0
        }
