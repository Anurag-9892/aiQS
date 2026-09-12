"""
RAG Engine orchestrating document indexing, embedding generation via Gemini API,
semantic retrieval, and grounded response generation.
"""

import os
from typing import List, Tuple, Generator, Optional, Dict, Any
from google import genai
from google.genai import types
from document_loader import DocumentChunk
from vector_store import VectorStore


from dotenv import load_dotenv
load_dotenv()

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_LLM_MODEL = "gemini-3.7-flash"



class RAGEngine:
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY") if api_key is None else api_key
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.vector_store = VectorStore()



    def set_api_key(self, api_key: str):
        """Update or set Gemini API key dynamically."""
        self.api_key = api_key.strip()
        self.client = genai.Client(api_key=self.api_key)

    def is_configured(self) -> bool:
        """Check if client is initialized with an API key."""
        return self.client is not None and bool(self.api_key)

    def embed_texts(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        """
        Compute vector embeddings for a list of text strings in batches.
        """
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured. Please provide an API key.")

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.models.embed_content(
                model=DEFAULT_EMBEDDING_MODEL,
                contents=batch,
            )
            # Response contains embeddings list
            if hasattr(response, "embeddings") and response.embeddings:
                for item in response.embeddings:
                    embeddings.append(list(item.values))
            elif hasattr(response, "embedding") and response.embedding:
                embeddings.append(list(response.embedding.values))
            else:
                raise RuntimeError(f"Unexpected response format from embedding API: {response}")

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Compute embedding vector for a single search query."""
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured. Please provide an API key.")

        response = self.client.models.embed_content(
            model=DEFAULT_EMBEDDING_MODEL,
            contents=query,
        )
        if hasattr(response, "embeddings") and response.embeddings:
            return list(response.embeddings[0].values)
        elif hasattr(response, "embedding") and response.embedding:
            return list(response.embedding.values)
        else:
            raise RuntimeError(f"Unexpected response format from embedding API: {response}")

    def index_chunks(self, chunks: List[DocumentChunk], session_id: Optional[str] = None) -> int:
        """
        Generate embeddings for document chunks, add them to vector store,
        and optionally persist to MongoDB.
        """
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        self.vector_store.add_chunks(chunks, embeddings)

        if session_id:
            try:
                from database import save_chunks_to_db
                save_chunks_to_db(session_id, chunks, embeddings)
            except Exception:
                pass

        return len(chunks)

    def ensure_loaded(self, session_id: str) -> None:
        """Load stored chunks from MongoDB if in-memory store is empty (serverless recovery)."""
        if not self.vector_store.chunks and session_id:
            try:
                from database import load_chunks_from_db
                chunks, embeddings = load_chunks_from_db(session_id)
                if chunks and embeddings:
                    self.vector_store.add_chunks(chunks, embeddings)
            except Exception:
                pass

    def retrieve_context(self, question: str, top_k: int = 4, session_id: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve top-k relevant document chunks for the user's question.
        Automatically checks MongoDB if in-memory store is empty.
        """
        if session_id:
            self.ensure_loaded(session_id)

        if not self.vector_store.chunks:
            return []

        query_vector = self.embed_query(query=question)
        return self.vector_store.similarity_search(query_embedding=query_vector, top_k=top_k)


    def generate_grounded_response_stream(
        self,
        question: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
        model_name: str = DEFAULT_LLM_MODEL
    ) -> Generator[str, None, None]:
        """
        Stream a grounded answer to the user's question based strictly on the retrieved chunks.
        """
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured. Please provide an API key.")

        # Build context section
        if not retrieved_chunks:
            yield "No relevant document excerpts were found in the index. Please make sure documents are uploaded and indexed."
            return

        context_blocks = []
        for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                f"[Source #{i} | File: {chunk.source_file} | Page {chunk.page_number} | Relevance: {score:.2f}]\n{chunk.text}"
            )
        context_str = "\n\n---\n\n".join(context_blocks)

        system_instruction = (
            "You are an expert Document Question-Answering Assistant (aiQS).\n"
            "Your task is to answer the user's question using ONLY the provided document context excerpts below.\n\n"
            "STRICT RULES:\n"
            "1. Answer truthfully and strictly according to the context provided.\n"
            "2. If the answer cannot be found in or deduced from the context, clearly state: "
            "'Based on the uploaded documents, I could not find information regarding this question.' "
            "Do NOT speculate, guess, or use external knowledge.\n"
            "3. Cite your sources directly in the answer using footnotes or inline citations like [File: <name>, Page <num>].\n"
            "4. Structure your answer cleanly with markdown (headings, bullet points, or tables where appropriate).\n"
        )

        user_prompt = (
            f"DOCUMENT CONTEXT EXCERPTS:\n"
            f"{context_str}\n\n"
            f"USER QUESTION:\n"
            f"{question}\n\n"
            f"Please provide a well-structured and accurate response with source citations:"
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,  # Low temperature for strict adherence and accuracy
        )

        response_stream = self.client.models.generate_content_stream(
            model=model_name,
            contents=user_prompt,
            config=config,
        )

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def clear_index(self):
        """Clear all stored documents from memory."""
        self.vector_store.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store summary statistics."""
        return self.vector_store.get_stats()
