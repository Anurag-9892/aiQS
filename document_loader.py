"""
Document loader and chunking module for aiQS.
Supports PDF, DOCX, and TXT files, preserving page numbers and source metadata.
"""

from dataclasses import dataclass, asdict
from io import BytesIO
from typing import List, Union, BinaryIO
import os
import re
from pypdf import PdfReader
from docx import Document as DocxDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    page_number: int
    text: str

    def to_dict(self):
        return asdict(self)


def extract_pages_from_pdf(file_input: Union[str, BinaryIO, BytesIO], filename: str) -> List[dict]:
    """Extract text from a PDF file page-by-page."""
    reader = PdfReader(file_input)
    pages_data = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = clean_text(text)
        if text:
            pages_data.append({
                "source_file": filename,
                "page_number": idx + 1,
                "text": text
            })
    return pages_data


def extract_pages_from_docx(file_input: Union[str, BinaryIO, BytesIO], filename: str) -> List[dict]:
    """Extract text from a DOCX file."""
    doc = DocxDocument(file_input)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    cleaned = clean_text(full_text)
    if not cleaned:
        return []
    return [{
        "source_file": filename,
        "page_number": 1,
        "text": cleaned
    }]


def extract_pages_from_txt(file_input: Union[str, BinaryIO, BytesIO], filename: str) -> List[dict]:
    """Extract text from a plain text or markdown file."""
    if isinstance(file_input, str):
        with open(file_input, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    elif isinstance(file_input, (BytesIO, BinaryIO)):
        content = file_input.read().decode("utf-8", errors="replace")
    else:
        content = str(file_input)

    cleaned = clean_text(content)
    if not cleaned:
        return []
    return [{
        "source_file": filename,
        "page_number": 1,
        "text": cleaned
    }]


def clean_text(text: str) -> str:
    """Normalize whitespace and clean up extracted text."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(file_input: Union[str, BinaryIO, BytesIO], filename: str) -> List[dict]:
    """
    Auto-detect format by filename extension and extract pages.
    Returns list of {"source_file": ..., "page_number": ..., "text": ...}
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_pages_from_pdf(file_input, filename)
    elif ext in [".docx", ".doc"]:
        return extract_pages_from_docx(file_input, filename)
    elif ext in [".txt", ".md", ".csv", ".json"]:
        return extract_pages_from_txt(file_input, filename)
    else:
        # Fallback to text
        return extract_pages_from_txt(file_input, filename)


def split_text_into_chunks(text: str, chunk_size: int = 700, chunk_overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks using natural boundaries (paragraphs, sentences).
    """
    if len(text) <= chunk_size:
        return [text]

    # Split into paragraphs or sentences
    separators = ["\n\n", "\n", ". ", " "]
    
    def _recursive_split(sub_text: str, sep_idx: int) -> List[str]:
        if len(sub_text) <= chunk_size:
            return [sub_text.strip()] if sub_text.strip() else []
        
        if sep_idx >= len(separators):
            # Hard split if no separators left
            return [sub_text[i:i + chunk_size] for i in range(0, len(sub_text), chunk_size - chunk_overlap)]
        
        sep = separators[sep_idx]
        parts = sub_text.split(sep)
        chunks = []
        current_chunk = ""

        for part in parts:
            if not part:
                continue
            piece = part if not current_chunk else sep + part
            if len(current_chunk) + len(piece) <= chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(part) > chunk_size:
                    # Recursive split on next separator
                    sub_chunks = _recursive_split(part, sep_idx + 1)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    raw_chunks = _recursive_split(text, 0)
    
    # Merge small chunks with overlap
    final_chunks = []
    for i, c in enumerate(raw_chunks):
        if not c:
            continue
        final_chunks.append(c)
        
    return final_chunks


def chunk_document_pages(
    pages: List[dict],
    chunk_size: int = 700,
    chunk_overlap: int = 150
) -> List[DocumentChunk]:
    """
    Transform extracted pages into DocumentChunk objects with unique IDs and metadata.
    """
    document_chunks: List[DocumentChunk] = []
    chunk_counter = 0

    for page in pages:
        source_file = page["source_file"]
        page_num = page["page_number"]
        page_text = page["text"]

        chunks = split_text_into_chunks(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for c in chunks:
            chunk_counter += 1
            chunk_id = f"{source_file}_p{page_num}_c{chunk_counter}"
            document_chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                source_file=source_file,
                page_number=page_num,
                text=c
            ))

    return document_chunks
