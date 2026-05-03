# =============================================================
# ResQAI — RAG Pipeline Module
# Handles PDF ingestion, chunking, embedding, and retrieval
# =============================================================

import os
import pickle
import hashlib
import numpy as np
from typing import List, Tuple, Optional

import faiss
import google.generativeai as genai
from pypdf import PdfReader

# Add parent to path for config import
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import (
    GEMINI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP,
    MAX_RETRIEVAL_DOCS, EMBEDDING_MODEL
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ─── Constants ────────────────────────────────────────────
FAISS_INDEX_PATH = "faiss_index.pkl"
FALLBACK_MESSAGE = (
    "I could not find this information in the uploaded disaster "
    "management documents. Please upload relevant PDFs or consult "
    "official emergency management resources."
)


# ─── Text Extraction ──────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract raw text from an uploaded PDF file object.

    Args:
        pdf_file: Streamlit UploadedFile or file-like object

    Returns:
        Extracted text as a single string
    """
    try:
        reader = PdfReader(pdf_file)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Add page marker for traceability
                pages_text.append(f"[Page {i+1}]\n{text.strip()}")
        return "\n\n".join(pages_text)
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


# ─── Text Chunking ────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.

    Args:
        text: Raw text to split
        chunk_size: Number of characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to break at a sentence or paragraph boundary
        if end < text_length:
            # Look for paragraph break first
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + (chunk_size // 2):
                end = para_break
            else:
                # Look for sentence boundary
                sent_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end),
                )
                if sent_break > start + (chunk_size // 2):
                    end = sent_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward accounting for overlap
        start = max(start + 1, end - overlap)

    return chunks


# ─── Embedding Generation ─────────────────────────────────

def get_embedding(text: str) -> List[float]:
    """
    Generate embedding vector for a text string using Gemini.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as list of floats
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]


def get_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for a search query (different task type).

    Args:
        query: Search query text

    Returns:
        Embedding vector as list of floats
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    return result["embedding"]


# ─── FAISS Vector Store ───────────────────────────────────

class FAISSVectorStore:
    """
    FAISS-based vector store for document chunk storage and retrieval.
    Supports incremental addition of documents.
    """

    def __init__(self):
        self.index = None          # FAISS index
        self.chunks = []           # Stored text chunks
        self.metadata = []         # Chunk metadata (source, page, etc.)
        self.dimension = None      # Embedding dimension
        self.doc_hashes = set()    # Track ingested documents

    def _init_index(self, dimension: int):
        """Initialize FAISS index with given embedding dimension."""
        self.dimension = dimension
        # Using IndexFlatIP for cosine similarity (with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)

    def add_documents(self, chunks: List[str], source_name: str = "unknown"):
        """
        Generate embeddings and add chunks to the FAISS index.

        Args:
            chunks: List of text chunks to add
            source_name: Name of the source document
        """
        if not chunks:
            return

        embeddings = []
        valid_chunks = []
        valid_metadata = []

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
            try:
                embedding = get_embedding(chunk)
                embeddings.append(embedding)
                valid_chunks.append(chunk)
                valid_metadata.append({
                    "source": source_name,
                    "chunk_id": len(self.chunks) + len(valid_chunks) - 1
                })
            except Exception as e:
                print(f"Warning: Failed to embed chunk {i}: {e}")
                continue

        if not embeddings:
            return

        # Convert to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_np)

        # Initialize index on first call
        if self.index is None:
            self._init_index(embeddings_np.shape[1])

        # Add to FAISS
        self.index.add(embeddings_np)
        self.chunks.extend(valid_chunks)
        self.metadata.extend(valid_metadata)

    def search(self, query: str, k: int = MAX_RETRIEVAL_DOCS) -> List[Tuple[str, float, dict]]:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: Search query string
            k: Number of results to retrieve

        Returns:
            List of (chunk_text, similarity_score, metadata) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Get query embedding
        query_embedding = get_query_embedding(query)
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)

        # Search FAISS
        k_actual = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_np, k_actual)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((
                    self.chunks[idx],
                    float(score),
                    self.metadata[idx]
                ))

        return results

    def save(self, path: str = FAISS_INDEX_PATH):
        """Persist the vector store to disk."""
        store_data = {
            "chunks": self.chunks,
            "metadata": self.metadata,
            "dimension": self.dimension,
            "doc_hashes": self.doc_hashes,
        }
        # Save FAISS index separately
        if self.index is not None:
            faiss.write_index(self.index, path + ".faiss")
        with open(path, "wb") as f:
            pickle.dump(store_data, f)

    def load(self, path: str = FAISS_INDEX_PATH) -> bool:
        """
        Load vector store from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(path):
                return False
            with open(path, "rb") as f:
                store_data = pickle.load(f)
            self.chunks = store_data["chunks"]
            self.metadata = store_data["metadata"]
            self.dimension = store_data["dimension"]
            self.doc_hashes = store_data.get("doc_hashes", set())
            # Load FAISS index
            faiss_path = path + ".faiss"
            if os.path.exists(faiss_path):
                self.index = faiss.read_index(faiss_path)
            return True
        except Exception:
            return False

    def is_empty(self) -> bool:
        """Check if the vector store has any documents."""
        return self.index is None or self.index.ntotal == 0

    def document_count(self) -> int:
        """Return number of stored chunks."""
        return len(self.chunks)

    def clear(self):
        """Clear all stored documents."""
        self.index = None
        self.chunks = []
        self.metadata = []
        self.doc_hashes = set()


# ─── RAG Query Engine ─────────────────────────────────────

def build_rag_context(results: List[Tuple[str, float, dict]],
                       min_score: float = 0.3) -> str:
    """
    Build context string from retrieved chunks.

    Args:
        results: List of (chunk, score, metadata) tuples
        min_score: Minimum similarity score threshold

    Returns:
        Formatted context string for the LLM
    """
    filtered = [(chunk, score, meta) for chunk, score, meta in results
                if score >= min_score]

    if not filtered:
        return ""

    context_parts = []
    for i, (chunk, score, meta) in enumerate(filtered, 1):
        source = meta.get("source", "Unknown")
        context_parts.append(
            f"[Source {i}: {source}]\n{chunk}"
        )

    return "\n\n---\n\n".join(context_parts)


def compute_file_hash(file_content: bytes) -> str:
    """Compute SHA256 hash of file content for deduplication."""
    return hashlib.sha256(file_content).hexdigest()
