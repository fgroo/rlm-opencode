"""Lightweight document embeddings and semantic search for RLM-OpenCode.

Uses BM25 scoring with tag/title boosting — zero external dependencies.
Chunks documents, builds an inverted index, and ranks by relevance.

This is the Layer 2 "vector search pre-filter" that narrows down docs
before RLM tools do precise reading.

Key improvements over naive TF-IDF:
- BM25 scoring with document length normalization
- Tag/title token boosting (query "sort" boosts docs with "sort" in their tag)
- Document-level aggregation (best chunk per doc, not per chunk)
- NAME/SYNOPSIS section weighting (first chunk gets a bonus)
"""

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Storage alongside the docs registry
EMBEDDINGS_DB = Path.home() / ".local" / "share" / "rlm-opencode" / "docs" / "embeddings.db"

# Chunking parameters
CHUNK_SIZE = 500       # Characters per chunk
CHUNK_OVERLAP = 100    # Overlap between chunks

# BM25 parameters
BM25_K1 = 1.5          # Term frequency saturation
BM25_B = 0.75          # Length normalization weight

# Boosting weights
TAG_BOOST = 3.0        # Multiplier when query term appears in document tag
TITLE_BOOST = 2.5      # Multiplier when query term appears in first chunk (NAME section)
HEADER_CHUNK_BOOST = 1.5  # Bonus multiplier for chunk_index 0 (NAME/SYNOPSIS)


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
        'how', 'what', 'when', 'where', 'which', 'who', 'why',
        'use', 'using', 'used', 'std', 'cpp', 'c__',
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def _extract_tag_tokens(tag: str) -> set[str]:
    """Extract searchable tokens from a document tag.
    
    e.g. "cppman/algorithm/std_sort" → {"sort", "algorithm", "std_sort"}
    """
    # Split on / and _
    parts = re.findall(r'[a-z0-9]+', tag.lower())
    # Also include compound names (std_sort → std_sort, sort)
    tokens = set(parts)
    # Add the full name segments from the last path component
    last_part = tag.rsplit('/', 1)[-1] if '/' in tag else tag
    tokens.update(re.findall(r'[a-z0-9_]+', last_part.lower()))
    # Remove common prefixes that add noise
    tokens.discard('std')
    tokens.discard('cppman')
    return {t for t in tokens if len(t) > 1}


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
    """BM25-based document search engine with tag boosting.
    
    Uses SQLite for persistence and a simple inverted index for search.
    No external dependencies required.
    
    Scoring: BM25(chunk) * tag_boost * position_boost, then
    document-level aggregation picks the best chunk per doc.
    """

    def __init__(self, db_path: Path = EMBEDDINGS_DB):
        self.db_path = db_path
        self._idf_cache: dict[str, float] = {}
        self._total_chunks: int = 0
        self._avg_doc_len: float = 0.0
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
                    token_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(tag, chunk_index)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS term_freq (
                    chunk_id INTEGER NOT NULL,
                    term TEXT NOT NULL,
                    raw_count INTEGER NOT NULL DEFAULT 0,
                    freq REAL NOT NULL,
                    PRIMARY KEY (chunk_id, term),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS doc_tags (
                    tag TEXT PRIMARY KEY,
                    tag_tokens_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_term ON term_freq(term)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_tag ON chunks(tag)
            """)
            # Migration: add columns if they don't exist (for existing DBs)
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE term_freq ADD COLUMN raw_count INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass

    def _get_db(self):
        return sqlite3.connect(str(self.db_path))

    def index_document(self, tag: str, content: str):
        """Chunk and index a document for semantic search.
        
        Args:
            tag: Document tag (e.g., "cppman/algorithm/std_sort")
            content: Full document text
        """
        tag_tokens = _extract_tag_tokens(tag)
        raw_chunks = _chunk_text(content)
        
        with self._get_db() as conn:
            # Remove old data for this tag
            old_ids = [row[0] for row in conn.execute(
                "SELECT id FROM chunks WHERE tag = ?", (tag,)
            ).fetchall()]
            if old_ids:
                placeholders = ",".join("?" * len(old_ids))
                conn.execute(f"DELETE FROM term_freq WHERE chunk_id IN ({placeholders})", old_ids)
                conn.execute("DELETE FROM chunks WHERE tag = ?", (tag,))
            
            # Store tag tokens
            conn.execute(
                "INSERT OR REPLACE INTO doc_tags (tag, tag_tokens_json) VALUES (?, ?)",
                (tag, json.dumps(list(tag_tokens)))
            )
            
            # Batch insert chunks and term frequencies
            term_freq_batch = []
            
            for i, (chunk_text, offset) in enumerate(raw_chunks):
                if not chunk_text.strip():
                    continue
                
                tokens = _tokenize(chunk_text)
                if not tokens:
                    continue
                
                token_counts = Counter(tokens)
                
                cursor = conn.execute(
                    "INSERT INTO chunks (tag, chunk_index, content, offset, length, tokens_json, token_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (tag, i, chunk_text, offset, len(chunk_text), json.dumps(tokens), len(tokens))
                )
                chunk_id = cursor.lastrowid
                
                max_count = max(token_counts.values()) if token_counts else 1
                for term, count in token_counts.items():
                    tf = 0.5 + 0.5 * (count / max_count)
                    term_freq_batch.append((chunk_id, term, count, tf))
            
            # Batch insert all term frequencies at once
            if term_freq_batch:
                conn.executemany(
                    "INSERT INTO term_freq (chunk_id, term, raw_count, freq) VALUES (?, ?, ?, ?)",
                    term_freq_batch
                )
        
        # Invalidate caches
        self._idf_cache.clear()
        self._total_chunks = 0
        self._avg_doc_len = 0.0

    def batch_index_documents(self, docs: list[tuple[str, str]]):
        """Index multiple documents in a single transaction.
        
        Much faster than calling index_document() in a loop.
        
        Args:
            docs: List of (tag, content) tuples
        """
        with self._get_db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            for tag, content in docs:
                tag_tokens = _extract_tag_tokens(tag)
                raw_chunks = _chunk_text(content)
                
                # Remove old data
                old_ids = [row[0] for row in conn.execute(
                    "SELECT id FROM chunks WHERE tag = ?", (tag,)
                ).fetchall()]
                if old_ids:
                    placeholders = ",".join("?" * len(old_ids))
                    conn.execute(f"DELETE FROM term_freq WHERE chunk_id IN ({placeholders})", old_ids)
                    conn.execute("DELETE FROM chunks WHERE tag = ?", (tag,))
                
                conn.execute(
                    "INSERT OR REPLACE INTO doc_tags (tag, tag_tokens_json) VALUES (?, ?)",
                    (tag, json.dumps(list(tag_tokens)))
                )
                
                term_freq_batch = []
                for i, (chunk_text, offset) in enumerate(raw_chunks):
                    if not chunk_text.strip():
                        continue
                    tokens = _tokenize(chunk_text)
                    if not tokens:
                        continue
                    
                    token_counts = Counter(tokens)
                    cursor = conn.execute(
                        "INSERT INTO chunks (tag, chunk_index, content, offset, length, tokens_json, token_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (tag, i, chunk_text, offset, len(chunk_text), json.dumps(tokens), len(tokens))
                    )
                    chunk_id = cursor.lastrowid
                    max_count = max(token_counts.values()) if token_counts else 1
                    for term, count in token_counts.items():
                        tf = 0.5 + 0.5 * (count / max_count)
                        term_freq_batch.append((chunk_id, term, count, tf))
                
                if term_freq_batch:
                    conn.executemany(
                        "INSERT INTO term_freq (chunk_id, term, raw_count, freq) VALUES (?, ?, ?, ?)",
                        term_freq_batch
                    )
        
        self._idf_cache.clear()
        self._total_chunks = 0
        self._avg_doc_len = 0.0

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
            conn.execute("DELETE FROM doc_tags WHERE tag = ?", (tag,))
        self._idf_cache.clear()
        self._total_chunks = 0
        self._avg_doc_len = 0.0

    def _ensure_stats(self, conn):
        """Load corpus-level stats."""
        if self._total_chunks == 0:
            row = conn.execute("SELECT COUNT(*), COALESCE(AVG(token_count), 1) FROM chunks").fetchone()
            self._total_chunks = row[0] if row[0] else 1
            self._avg_doc_len = row[1] if row[1] else 1.0

    def _get_idf(self, term: str, conn) -> float:
        """Get Inverse Document Frequency for a term (BM25 style)."""
        if term in self._idf_cache:
            return self._idf_cache[term]
        
        self._ensure_stats(conn)
        
        row = conn.execute(
            "SELECT COUNT(DISTINCT chunk_id) FROM term_freq WHERE term = ?", (term,)
        ).fetchone()
        doc_freq = row[0] if row else 0
        
        # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        idf = math.log((self._total_chunks - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        self._idf_cache[term] = idf
        return idf

    def _get_tag_tokens(self, tag: str, conn) -> set[str]:
        """Get cached tag tokens for a document."""
        row = conn.execute("SELECT tag_tokens_json FROM doc_tags WHERE tag = ?", (tag,)).fetchone()
        if row:
            return set(json.loads(row[0]))
        return _extract_tag_tokens(tag)

    def search(self, query: str, top_k: int = 5, tag_filter: str | None = None) -> list[SearchResult]:
        """Search indexed documents using BM25 with tag boosting.
        
        Scoring pipeline:
        1. BM25 score per chunk (term frequency + length normalization)
        2. Tag boost: query terms matching document tag get multiplied
        3. Position boost: first chunk (NAME/SYNOPSIS) gets a bonus
        4. Document aggregation: keep only best chunk per document
        
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
            self._ensure_stats(conn)
            
            # Find candidate chunks
            terms_placeholder = ",".join("?" * len(query_tokens))
            
            if tag_filter:
                candidates = conn.execute(f"""
                    SELECT DISTINCT c.id, c.tag, c.chunk_index, c.content, c.offset, c.length, c.token_count
                    FROM chunks c
                    JOIN term_freq tf ON c.id = tf.chunk_id
                    WHERE tf.term IN ({terms_placeholder})
                    AND c.tag LIKE ?
                """, (*query_tokens, f"{tag_filter}%")).fetchall()
            else:
                candidates = conn.execute(f"""
                    SELECT DISTINCT c.id, c.tag, c.chunk_index, c.content, c.offset, c.length, c.token_count
                    FROM chunks c
                    JOIN term_freq tf ON c.id = tf.chunk_id
                    WHERE tf.term IN ({terms_placeholder})
                """, query_tokens).fetchall()
            
            if not candidates:
                return []
            
            # Pre-fetch tag tokens for all unique tags
            unique_tags = set(row[1] for row in candidates)
            tag_tokens_map: dict[str, set[str]] = {}
            for tag in unique_tags:
                tag_tokens_map[tag] = self._get_tag_tokens(tag, conn)
            
            # Score each candidate using BM25
            chunk_scores: list[tuple[float, int, str, int, str, int, int]] = []
            
            for chunk_id, tag, chunk_idx, content, offset, length, token_count in candidates:
                doc_len = token_count if token_count > 0 else len(_tokenize(content))
                
                # BM25 score for this chunk
                bm25_score = 0.0
                matched_terms = 0
                
                for term in set(query_tokens):  # deduplicate query terms
                    idf = self._get_idf(term, conn)
                    
                    row = conn.execute(
                        "SELECT raw_count FROM term_freq WHERE chunk_id = ? AND term = ?",
                        (chunk_id, term)
                    ).fetchone()
                    
                    if row:
                        raw_tf = row[0]
                        matched_terms += 1
                        
                        # BM25 TF component
                        numerator = raw_tf * (BM25_K1 + 1)
                        denominator = raw_tf + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / self._avg_doc_len))
                        bm25_score += idf * (numerator / denominator)
                
                if bm25_score <= 0:
                    continue
                
                # --- Boosting ---
                
                # Tag boost: if query terms match the tag, multiply the score
                tag_tokens = tag_tokens_map.get(tag, set())
                tag_match_count = sum(1 for t in set(query_tokens) if t in tag_tokens)
                if tag_match_count > 0:
                    bm25_score *= (1 + TAG_BOOST * tag_match_count / len(set(query_tokens)))
                
                # Position boost: NAME/SYNOPSIS chunks (index 0-1) are more important
                if chunk_idx == 0:
                    bm25_score *= HEADER_CHUNK_BOOST
                elif chunk_idx == 1:
                    bm25_score *= 1.2
                
                # Coverage bonus: more query terms matched → higher score
                coverage = matched_terms / len(set(query_tokens))
                bm25_score *= (0.5 + 0.5 * coverage)
                
                chunk_scores.append((bm25_score, chunk_id, tag, chunk_idx, content, offset, length))
            
            # Document-level aggregation: keep best chunk per tag
            best_per_doc: dict[str, tuple[float, int, str, int, str, int, int]] = {}
            for entry in chunk_scores:
                score, chunk_id, tag, chunk_idx, content, offset, length = entry
                if tag not in best_per_doc or score > best_per_doc[tag][0]:
                    best_per_doc[tag] = entry
            
            # Sort by score and return top_k
            sorted_results = sorted(best_per_doc.values(), key=lambda x: x[0], reverse=True)
            
            # Normalize scores to 0-1 range for readability
            if sorted_results:
                max_score = sorted_results[0][0]
                min_score = sorted_results[-1][0] if len(sorted_results) > 1 else 0
                score_range = max_score - min_score if max_score > min_score else 1.0
            
            results = []
            for score, chunk_id, tag, chunk_idx, content, offset, length in sorted_results[:top_k]:
                # Normalize to 0-1
                normalized = (score - min_score) / score_range if score_range > 0 else 1.0
                normalized = max(0.0, min(1.0, normalized))
                
                results.append(SearchResult(
                    tag=tag,
                    chunk_index=chunk_idx,
                    content=content[:300],
                    score=round(normalized, 4),
                    offset=offset,
                    length=length,
                ))
            
            return results

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
