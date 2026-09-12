"""
MongoDB database module for aiQS.
Handles persistent storage of:
- Chat history (conversations & messages)
- Document index metadata (uploaded files, chunk stats)
- Session data
"""

import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


# ─── Connection ────────────────────────────────────────────────────────────────

MONGO_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://anurag04885_db_user:2E9FR2fjClc5z4YU@cluster0.dhzlh98.mongodb.net/?appName=Cluster0"
)
DATABASE_NAME = "aiqs_db"

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_db() -> Database:
    """Get (or create) the MongoDB database connection."""
    global _client, _db
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _db = _client[DATABASE_NAME]
    return _db


def ping() -> bool:
    """Test the database connection. Returns True if reachable."""
    try:
        get_db().command("ping")
        return True
    except Exception:
        return False


# ─── Collections ───────────────────────────────────────────────────────────────

def chat_collection() -> Collection:
    return get_db()["chat_history"]


def documents_collection() -> Collection:
    return get_db()["documents"]


def sessions_collection() -> Collection:
    return get_db()["sessions"]


def chunks_collection() -> Collection:
    return get_db()["document_chunks"]


# ─── Chunk & Vector Storage ───────────────────────────────────────────────────

def save_chunks_to_db(session_id: str, chunks: list, embeddings: list) -> int:
    """Persist document chunks and their embedding vectors to MongoDB."""
    if not chunks or not embeddings or len(chunks) != len(embeddings):
        return 0

    docs = []
    for chunk, emb in zip(chunks, embeddings):
        docs.append({
            "session_id": session_id,
            "chunk_id": chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk["chunk_id"],
            "source_file": chunk.source_file if hasattr(chunk, "source_file") else chunk["source_file"],
            "page_number": chunk.page_number if hasattr(chunk, "page_number") else chunk["page_number"],
            "text": chunk.text if hasattr(chunk, "text") else chunk["text"],
            "embedding": emb,
            "created_at": datetime.now(timezone.utc),
        })

    result = chunks_collection().insert_many(docs)
    return len(result.inserted_ids)


def load_chunks_from_db(session_id: str):
    """Load all chunks and vectors for a given session from MongoDB."""
    from document_loader import DocumentChunk
    cursor = chunks_collection().find({"session_id": session_id})
    chunks = []
    embeddings = []
    for doc in cursor:
        chunks.append(DocumentChunk(
            chunk_id=doc["chunk_id"],
            source_file=doc["source_file"],
            page_number=doc["page_number"],
            text=doc["text"],
        ))
        embeddings.append(doc["embedding"])
    return chunks, embeddings


def delete_chunks_for_session(session_id: str) -> int:
    """Delete all chunks and vectors for a session."""
    result = chunks_collection().delete_many({"session_id": session_id})
    return result.deleted_count



# ─── Chat History ──────────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str, citations: Optional[List[Dict]] = None) -> str:
    """Save a single chat message and return its inserted ID."""
    doc = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "citations": citations or [],
        "created_at": datetime.now(timezone.utc),
    }
    result = chat_collection().insert_one(doc)
    return str(result.inserted_id)


def get_chat_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve chat messages for a given session, ordered oldest-first."""
    cursor = (
        chat_collection()
        .find({"session_id": session_id}, {"_id": 0})
        .sort("created_at", 1)
        .limit(limit)
    )
    return list(cursor)


def delete_chat_history(session_id: str) -> int:
    """Delete all chat messages for a session. Returns deleted count."""
    result = chat_collection().delete_many({"session_id": session_id})
    return result.deleted_count


# ─── Document Metadata ─────────────────────────────────────────────────────────

def save_document_meta(session_id: str, filename: str, pages: int, chunks: int) -> str:
    """Record metadata about a successfully indexed document."""
    doc = {
        "session_id": session_id,
        "filename": filename,
        "pages": pages,
        "chunks": chunks,
        "indexed_at": datetime.now(timezone.utc),
    }
    result = documents_collection().insert_one(doc)
    return str(result.inserted_id)


def get_documents_for_session(session_id: str) -> List[Dict[str, Any]]:
    """Get all documents indexed for a given session."""
    cursor = documents_collection().find(
        {"session_id": session_id}, {"_id": 0}
    ).sort("indexed_at", -1)
    return list(cursor)


def delete_documents_for_session(session_id: str) -> int:
    """Remove all document metadata for a session."""
    result = documents_collection().delete_many({"session_id": session_id})
    return result.deleted_count


# ─── Session Registry ──────────────────────────────────────────────────────────

def create_or_update_session(session_id: str, extra: Optional[Dict] = None) -> None:
    """Upsert a session record with last_seen timestamp."""
    sessions_collection().update_one(
        {"session_id": session_id},
        {
            "$set": {
                "last_seen": datetime.now(timezone.utc),
                **(extra or {}),
            },
            "$setOnInsert": {
                "created_at": datetime.now(timezone.utc),
                "session_id": session_id,
            },
        },
        upsert=True,
    )


def get_all_sessions() -> List[Dict[str, Any]]:
    """Return all registered sessions (admin view)."""
    cursor = sessions_collection().find({}, {"_id": 0}).sort("last_seen", -1)
    return list(cursor)
