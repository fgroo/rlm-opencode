"""Document Registry for RLM-OpenCode.

Manages external documentation references that can be loaded into session context.
Documents are registered with tags and stored in a JSON index file.

Tags follow the format: category/name (e.g., cppman/time, react/hooks, fastapi/routing)
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Where doc registry lives
RLM_DOCS_DIR = Path.home() / ".local" / "share" / "rlm-opencode" / "docs"
DOCS_INDEX_FILE = RLM_DOCS_DIR / "docs_index.json"


@dataclass
class DocEntry:
    """A registered document."""
    tag: str                    # e.g., "cppman/time"
    file_path: str              # Absolute path to the source file
    title: str                  # Human-readable title
    size_chars: int = 0         # Size of document content
    doc_type: str = "text"      # text, markdown, manpage
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "file_path": self.file_path,
            "title": self.title,
            "size_chars": self.size_chars,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocEntry":
        return cls(**data)


class DocsRegistry:
    """Registry of external documents that can be loaded into RLM context."""

    def __init__(self, index_path: Path = DOCS_INDEX_FILE):
        self.index_path = index_path
        self._entries: dict[str, DocEntry] = {}
        self._load_index()

    def _load_index(self):
        """Load the document index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    data = json.load(f)
                self._entries = {
                    tag: DocEntry.from_dict(entry)
                    for tag, entry in data.get("docs", {}).items()
                }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load docs index: {e}")
                self._entries = {}
        else:
            self._entries = {}

    def _save_index(self):
        """Save the document index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "docs": {tag: entry.to_dict() for tag, entry in self._entries.items()}
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, tag: str, file_path: str, title: str | None = None) -> DocEntry:
        """Register a document with a tag.
        
        Args:
            tag: Document tag (e.g., "cppman/time", "react/hooks")
            file_path: Path to the document file
            title: Optional human-readable title (defaults to tag)
        
        Returns:
            The created DocEntry
        """
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Detect doc type from extension
        ext = path.suffix.lower()
        doc_type_map = {
            ".md": "markdown",
            ".txt": "text",
            ".man": "manpage",
            ".1": "manpage", ".2": "manpage", ".3": "manpage",
            ".html": "html",
            ".rst": "text",
        }
        doc_type = doc_type_map.get(ext, "text")

        content = self._read_file(path, doc_type)
        
        entry = DocEntry(
            tag=tag,
            file_path=str(path),
            title=title or tag,
            size_chars=len(content),
            doc_type=doc_type,
        )

        self._entries[tag] = entry
        self._save_index()
        return entry

    def remove(self, tag: str) -> bool:
        """Remove a document by tag. Returns True if found and removed."""
        if tag in self._entries:
            del self._entries[tag]
            self._save_index()
            return True
        return False

    def get(self, tag: str) -> DocEntry | None:
        """Get a document entry by tag."""
        return self._entries.get(tag)

    def get_content(self, tag: str) -> str | None:
        """Load and return the content of a registered document."""
        entry = self._entries.get(tag)
        if not entry:
            return None
        
        path = Path(entry.file_path)
        if not path.exists():
            return f"[ERROR: Document file missing: {entry.file_path}]"
        
        return self._read_file(path, entry.doc_type)

    def list_docs(self) -> list[DocEntry]:
        """List all registered documents."""
        return list(self._entries.values())

    def search_tags(self, query: str) -> list[DocEntry]:
        """Search documents by tag or title keyword."""
        query_lower = query.lower()
        results = []
        for entry in self._entries.values():
            if query_lower in entry.tag.lower() or query_lower in entry.title.lower():
                results.append(entry)
        return results

    def import_directory(self, directory: str, prefix: str = "") -> list[DocEntry]:
        """Bulk-register all docs in a directory.
        
        Args:
            directory: Path to directory containing doc files
            prefix: Tag prefix (e.g., "fastapi" → tags become "fastapi/filename")
        
        Returns:
            List of created DocEntries
        """
        dir_path = Path(directory).expanduser().resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        supported_exts = {".md", ".txt", ".man", ".rst", ".html", ".1", ".2", ".3"}
        entries = []

        for file in sorted(dir_path.rglob("*")):
            if file.is_file() and file.suffix.lower() in supported_exts:
                # Build tag from relative path
                rel = file.relative_to(dir_path)
                tag_parts = list(rel.parts[:-1]) + [rel.stem]
                tag = "/".join(tag_parts)
                if prefix:
                    tag = f"{prefix}/{tag}"
                
                try:
                    entry = self.add(tag, str(file), title=f"{prefix or 'doc'}: {rel.stem}")
                    entries.append(entry)
                except Exception as e:
                    print(f"Warning: Could not register {file}: {e}")

        return entries

    @staticmethod
    def _read_file(path: Path, doc_type: str) -> str:
        """Read a file and optionally strip HTML."""
        content = path.read_text(errors="replace")

        if doc_type == "html":
            # Simple HTML stripping (no dependency needed)
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
            content = re.sub(r'<[^>]+>', '', content)
            content = re.sub(r'\s+', ' ', content).strip()

        return content

    def format_doc_for_context(self, tag: str) -> str | None:
        """Format a document for injection into session context.
        
        Wraps the content with [DOCS: tag] labels so rlm_search can find it.
        """
        entry = self.get(tag)
        content = self.get_content(tag)
        if not entry or not content:
            return None

        return (
            f"\n{'='*60}\n"
            f"[DOCS: {entry.tag}] — {entry.title}\n"
            f"{'='*60}\n"
            f"{content}\n"
            f"{'='*60}\n"
            f"[END DOCS: {entry.tag}]\n"
        )


# Global singleton
_docs_registry: DocsRegistry | None = None

def get_docs_registry() -> DocsRegistry:
    """Get or create the global docs registry."""
    global _docs_registry
    if _docs_registry is None:
        _docs_registry = DocsRegistry()
    return _docs_registry
