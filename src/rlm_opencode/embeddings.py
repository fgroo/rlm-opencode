"""Lightweight document embeddings and semantic search for RLM-OpenCode.

Uses TF-IDF with cosine similarity — zero external dependencies.
Chunks documents, builds an inverted index, and ranks by relevance.

This is the Layer 2 "vector search pre-filter" that narrows down docs
before RLM tools do precise reading.
"""

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Storage alongside the docs registry
EMBEDDINGS_DB = Path.home() / ".local" / "share" / "rlm-opencode" / "docs" / "embeddings.db"

# Chunking parameters
CHUNK_SIZE = 500       # Characters per chunk
CHUNK_OVERLAP = 100    # Overlap between chunks


@dataclass
class DocChunk:
    """A chunk of a document with its position."""
    tag: str
    chunk_index: int
    content: str
    offset: int      # Character offset in original document
    length: int


@dataclass
class SearchResult:
    """A search result with relevance score."""
    tag: str
    chunk_index: int
    content: str
    score: float
    offset: int
    length: int


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r'[a-z0-9_]+', text.lower())
    # Minimal stopword list
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
        'should', 'may', 'might', 'must', 'can', 'could', 'of', 'in', 'to',
        'for', 'with', 'on', 'at', 'from', 'by', 'as', 'or', 'and', 'but',
        'if', 'not', 'no', 'so', 'it', 'its', 'this', 'that', 'these',
        'those', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my',
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[str, int]]:
    """Split text into overlapping chunks. Returns list of (chunk_text, offset)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at a newline or sentence boundary
        if end < len(text):
            last_newline = chunk.rfind('\n')
            if last_newline > chunk_size // 2:
                end = start + last_newline + 1
                chunk = text[start:end]
        
        chunks.append((chunk.strip(), start))
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


class DocEmbeddings:
    """TF-IDF based document search engine.
    
    Uses SQLite for persistence and a simple inverted index for search.
    No external dependencies required.
    """

    def __init__(self, db_path: Path = EMBEDDINGS_DB):
        self.db_path = db_path
        self._idf_cache: dict[str, float] = {}
        self._total_chunks: int = 0
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database for chunk storage."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    offset INTEGER NOT NULL,
                    length INTEGER NOT NULL,
                    tokens_json TEXT NOT NULL,
                    UNIQUE(tag, chunk_index)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS term_freq (
                    chunk_id INTEGER NOT NULL,
                    term TEXT NOT NULL,
                    freq REAL NOT NULL,
                    PRIMARY KEY (chunk_id, term),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_term ON term_freq(term)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_tag ON chunks(tag)
            """)

    def _get_db(self):
        return sqlite3.connect(str(self.db_path))

    def index_document(self, tag: str, content: str):
        """Chunk and index a document for semantic search.
        
        Args:
            tag: Document tag (e.g., "cppman/time")
            content: Full document text
        """
        # Remove old chunks for this tag
        with self._get_db() as conn:
            # Get old chunk IDs to delete term_freq entries
            old_ids = [row[0] for row in conn.execute(
                "SELECT id FROM chunks WHERE tag = ?", (tag,)
            ).fetchall()]
            if old_ids:
                placeholders = ",".join("?" * len(old_ids))
                conn.execute(f"DELETE FROM term_freq WHERE chunk_id IN ({placeholders})", old_ids)
                conn.execute("DELETE FROM chunks WHERE tag = ?", (tag,))

        # Chunk the content
        raw_chunks = _chunk_text(content)
        
        with self._get_db() as conn:
            for i, (chunk_text, offset) in enumerate(raw_chunks):
                if not chunk_text.strip():
                    continue
                
                tokens = _tokenize(chunk_text)
                if not tokens:
                    continue
                
                # Calculate TF (term frequency) for this chunk
                token_counts = Counter(tokens)
                max_count = max(token_counts.values()) if token_counts else 1
                
                cursor = conn.execute("""
                    INSERT INTO chunks (tag, chunk_index, content, offset, length, tokens_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (tag, i, chunk_text, offset, len(chunk_text), json.dumps(tokens)))
                
                chunk_id = cursor.lastrowid
                
                for term, count in token_counts.items():
                    tf = 0.5 + 0.5 * (count / max_count)  # Augmented TF
                    conn.execute("""
                        INSERT INTO term_freq (chunk_id, term, freq)
                        VALUES (?, ?, ?)
                    """, (chunk_id, term, tf))
        
        # Invalidate IDF cache
        self._idf_cache.clear()
        self._total_chunks = 0

    def remove_document(self, tag: str):
        """Remove all indexed chunks for a document."""
        with self._get_db() as conn:
            old_ids = [row[0] for row in conn.execute(
                "SELECT id FROM chunks WHERE tag = ?", (tag,)
            ).fetchall()]
            if old_ids:
                placeholders = ",".join("?" * len(old_ids))
                conn.execute(f"DELETE FROM term_freq WHERE chunk_id IN ({placeholders})", old_ids)
            conn.execute("DELETE FROM chunks WHERE tag = ?", (tag,))
        self._idf_cache.clear()
        self._total_chunks = 0

    def _get_idf(self, term: str, conn) -> float:
        """Get Inverse Document Frequency for a term."""
        if term in self._idf_cache:
            return self._idf_cache[term]
        
        if self._total_chunks == 0:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            self._total_chunks = row[0] if row else 1
        
        row = conn.execute(
            "SELECT COUNT(DISTINCT chunk_id) FROM term_freq WHERE term = ?", (term,)
        ).fetchone()
        doc_freq = row[0] if row else 0
        
        idf = math.log((self._total_chunks + 1) / (doc_freq + 1)) + 1
        self._idf_cache[term] = idf
        return idf

    def search(self, query: str, top_k: int = 5, tag_filter: str | None = None) -> list[SearchResult]:
        """Search indexed documents using TF-IDF cosine similarity.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            tag_filter: Optional tag prefix to filter results
        
        Returns:
            List of SearchResult ordered by relevance score
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        
        with self._get_db() as conn:
            # Build query vector (TF-IDF for each query term)
            query_tf = Counter(query_tokens)
            max_qf = max(query_tf.values())
            query_vector: dict[str, float] = {}
            for term, count in query_tf.items():
                tf = 0.5 + 0.5 * (count / max_qf)
                idf = self._get_idf(term, conn)
                query_vector[term] = tf * idf
            
            # Find candidate chunks (any chunk containing at least one query term)
            terms_placeholder = ",".join("?" * len(query_tokens))
            
            if tag_filter:
                candidates = conn.execute(f"""
                    SELECT DISTINCT c.id, c.tag, c.chunk_index, c.content, c.offset, c.length
                    FROM chunks c
                    JOIN term_freq tf ON c.id = tf.chunk_id
                    WHERE tf.term IN ({terms_placeholder})
                    AND c.tag LIKE ?
                """, (*query_tokens, f"{tag_filter}%")).fetchall()
            else:
                candidates = conn.execute(f"""
                    SELECT DISTINCT c.id, c.tag, c.chunk_index, c.content, c.offset, c.length
                    FROM chunks c
                    JOIN term_freq tf ON c.id = tf.chunk_id
                    WHERE tf.term IN ({terms_placeholder})
                """, query_tokens).fetchall()
            
            if not candidates:
                return []
            
            # Score each candidate using cosine similarity
            results: list[SearchResult] = []
            
            for chunk_id, tag, chunk_idx, content, offset, length in candidates:
                # Get chunk's TF-IDF vector (only for query terms)
                chunk_tfidf: dict[str, float] = {}
                for term in query_tokens:
                    row = conn.execute(
                        "SELECT freq FROM term_freq WHERE chunk_id = ? AND term = ?",
                        (chunk_id, term)
                    ).fetchone()
                    if row:
                        idf = self._get_idf(term, conn)
                        chunk_tfidf[term] = row[0] * idf
                
                # Cosine similarity
                dot_product = sum(
                    query_vector.get(t, 0) * chunk_tfidf.get(t, 0)
                    for t in query_tokens
                )
                query_norm = math.sqrt(sum(v ** 2 for v in query_vector.values()))
                chunk_norm = math.sqrt(sum(v ** 2 for v in chunk_tfidf.values())) if chunk_tfidf else 0
                
                if query_norm > 0 and chunk_norm > 0:
                    score = dot_product / (query_norm * chunk_norm)
                else:
                    score = 0.0
                
                results.append(SearchResult(
                    tag=tag,
                    chunk_index=chunk_idx,
                    content=content[:300],  # Preview
                    score=round(score, 4),
                    offset=offset,
                    length=length,
                ))
            
            # Sort by score descending, take top_k
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]

    def get_stats(self) -> dict:
        """Get indexing statistics."""
        with self._get_db() as conn:
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_terms = conn.execute("SELECT COUNT(DISTINCT term) FROM term_freq").fetchone()[0]
            tags = [row[0] for row in conn.execute("SELECT DISTINCT tag FROM chunks").fetchall()]
        
        return {
            "total_chunks": total_chunks,
            "total_terms": total_terms,
            "indexed_docs": len(tags),
            "tags": tags,
        }


# Global singleton
_embeddings: DocEmbeddings | None = None

def get_embeddings() -> DocEmbeddings:
    """Get or create the global embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = DocEmbeddings()
    return _embeddings
