"""Session management for RLM-OpenCode.

Enhanced session system with:
- Incognito mode (no persistence)
- OpenCode session syncing
- SQLite-based session database
- Efficient context storage

Schema:
- sessions.db: SQLite database for session metadata
- contexts/: Context files (one per session)
- mappings/: Session mappings (directory, opencode_id)
"""
import json
import sqlite3
import time
import uuid
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console

console = Console()

RLM_DATA_DIR = Path.home() / ".local" / "share" / "rlm-opencode"
SESSIONS_DB = RLM_DATA_DIR / "sessions.db"
CONTEXTS_DIR = RLM_DATA_DIR / "contexts"
MAPPINGS_DIR = RLM_DATA_DIR / "mappings"
CONFIG_FILE = RLM_DATA_DIR / "config.json"


@dataclass
class ContextEntry:
    """A single entry in the session context."""
    type: str
    offset: int
    length: int
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionStats:
    """Statistics about a session."""
    files_read: int = 0
    tool_outputs: int = 0
    thinking_blocks: int = 0
    total_chars: int = 0


@dataclass
class Session:
    """Represents an RLM-OpenCode session."""
    id: str
    created: float
    opencode_session_id: str | None = None
    directory: str | None = None
    incognito: bool = False
    target_session_id: str | None = None
    parent_session_id: str | None = None
    entries: list[ContextEntry] = field(default_factory=list)
    stats: SessionStats = field(default_factory=SessionStats)
    context_file: str = ""
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created": self.created,
            "opencode_session_id": self.opencode_session_id,
            "directory": self.directory,
            "incognito": self.incognito,
            "target_session_id": self.target_session_id,
            "parent_session_id": self.parent_session_id,
            "context_file": self.context_file,
            "stats": {
                "files_read": self.stats.files_read,
                "tool_outputs": self.stats.tool_outputs,
                "thinking_blocks": self.stats.thinking_blocks,
                "total_chars": self.stats.total_chars,
            },
            "entries": [
                {
                    "type": e.type,
                    "offset": e.offset,
                    "length": e.length,
                    "timestamp": e.timestamp,
                    "metadata": e.metadata,
                }
                for e in self.entries
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        stats = SessionStats(
            files_read=data.get("stats", {}).get("files_read", 0),
            tool_outputs=data.get("stats", {}).get("tool_outputs", 0),
            thinking_blocks=data.get("stats", {}).get("thinking_blocks", 0),
            total_chars=data.get("stats", {}).get("total_chars", 0),
        )
        entries = [
            ContextEntry(
                type=e["type"],
                offset=e["offset"],
                length=e["length"],
                timestamp=e["timestamp"],
                metadata=e.get("metadata", {}),
            )
            for e in data.get("entries", [])
        ]
        return cls(
            id=data["id"],
            created=data["created"],
            opencode_session_id=data.get("opencode_session_id"),
            directory=data.get("directory"),
            incognito=data.get("incognito", False),
            target_session_id=data.get("target_session_id"),
            parent_session_id=data.get("parent_session_id"),
            context_file=data.get("context_file", ""),
            entries=entries,
            stats=stats,
        )


class SessionManager:
    """Manages RLM-OpenCode sessions with SQLite backend.
    
    Features:
    - Incognito mode: Sessions that don't persist to disk
    - OpenCode syncing: Automatic sync with opencode session IDs
    - Directory mapping: Stable sessions per project directory
    - SQLite database: Fast queries and reliable storage
    """
    
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self._incognito_sessions: set[str] = set()
        self._incognito_contexts: dict[str, str] = {}
        self._locks = defaultdict(threading.Lock)
        self._ensure_dirs()
        self._init_db()
    
    def _ensure_dirs(self):
        """Ensure data directories exist."""
        RLM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
        MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Ensure default config exists
        if not CONFIG_FILE.exists():
            self._save_config({"strict_mode_level": 0})
            
    def _save_config(self, config_data: dict):
        """Save global configuration to disk."""
        CONFIG_FILE.write_text(json.dumps(config_data, indent=2))
        
    def get_config(self) -> dict:
        """Get global configuration."""
        if not CONFIG_FILE.exists():
            return {"strict_mode_level": 0}
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {"strict_mode_level": 0}
            
    def set_strict_mode(self, level: int):
        """Update the strict mode level (0-4)."""
        config = self.get_config()
        config["strict_mode_level"] = max(0, min(4, level))
        self._save_config(config)
    
    def _init_db(self):
        """Initialize SQLite database."""
        with self._get_db() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created REAL NOT NULL,
                    opencode_session_id TEXT,
                    directory TEXT,
                    incognito INTEGER DEFAULT 0,
                    target_session_id TEXT,
                    parent_session_id TEXT,
                    context_file TEXT,
                    stats_json TEXT,
                    UNIQUE(opencode_session_id)
                );
                
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    offset INTEGER NOT NULL,
                    length INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_sessions_opencode ON sessions(opencode_session_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_directory ON sessions(directory);
                CREATE INDEX IF NOT EXISTS idx_entries_session ON entries(session_id);
            """)
            
            # Migration to add target_session_id for existing DBs
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN target_session_id TEXT")
            except sqlite3.OperationalError:
                pass
                
            # Migration to add parent_session_id
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN parent_session_id TEXT")
            except sqlite3.OperationalError:
                pass

    
    @contextmanager
    def _get_db(self) -> Iterator[sqlite3.Connection]:
        """Get database connection."""
        conn = sqlite3.connect(str(SESSIONS_DB))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def create_session(
        self,
        opencode_session_id: str | None = None,
        directory: str | None = None,
        incognito: bool = False,
    ) -> Session:
        """Create a new session.
        
        Args:
            opencode_session_id: Optional opencode session to sync with
            directory: Optional working directory for stable mapping
            incognito: If True, session won't persist to disk
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        context_file = f"{session_id}_context.txt"
        
        session = Session(
            id=session_id,
            created=time.time(),
            opencode_session_id=opencode_session_id,
            directory=directory,
            incognito=incognito,
            context_file=context_file,
        )
        
        if not incognito:
            context_path = CONTEXTS_DIR / context_file
            context_path.touch()
        
        self.sessions[session_id] = session
        
        if incognito:
            self._incognito_sessions.add(session_id)
            self._incognito_contexts[session_id] = ""
            console.print(f"[dim]Created incognito session {session_id}[/dim]")
        else:
            self._save_session(session)
            console.print(f"[green]Created session {session_id}[/green]")
        
        return session
    
    def get_raw_session(self, session_id: str) -> Session | None:
        """Get a session by ID without resolving its target."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        session = self._load_session(session_id)
        if session:
            self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str, resolve: bool = True) -> Session | None:
        """Get a session by ID, optionally resolving target links."""
        session = self.get_raw_session(session_id)
        if not session or not resolve:
            return session
            
        depth = 0
        while session.target_session_id and depth < 5:
            target = self.get_raw_session(session.target_session_id)
            if not target:
                break
            session = target
            depth += 1
            
        return session

    def set_target_session(self, session_id: str, target_id: str | None) -> bool:
        """Link a session to another session's context."""
        session = self.get_raw_session(session_id)
        if not session:
            return False
            
        session.target_session_id = target_id
        if not session.incognito:
            with self._get_db() as conn:
                conn.execute(
                    "UPDATE sessions SET target_session_id = ? WHERE id = ?",
                    (target_id, session_id)
                )
        return True
    
    def get_session_by_opencode(self, opencode_session_id: str) -> Session | None:
        """Get or create session synced with an opencode session."""
        with self._get_db() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE opencode_session_id = ?",
                (opencode_session_id,)
            ).fetchone()
            
            if row:
                session = self._row_to_session(row)
                self.sessions[session.id] = session
                return self.get_session(session.id)
        
        return self.create_session(opencode_session_id=opencode_session_id)
    
    def get_or_create_session_by_directory(self, directory: str, incognito: bool = False) -> Session:
        """Get or create session by working directory."""
        dir_path = str(Path(directory).resolve())
        
        for session in self.sessions.values():
            if session.directory == dir_path:
                return self.get_session(session.id)
        
        with self._get_db() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE directory = ?",
                (dir_path,)
            ).fetchone()
            
            if row:
                session = self._row_to_session(row)
                self.sessions[session.id] = session
                return self.get_session(session.id)
        
        session = self.create_session(directory=dir_path, incognito=incognito)
        console.print(f"[green]Created session {session.id} for directory: {dir_path}[/green]")
        return session

    def get_or_create_session_by_opencode_id(self, opencode_id: str, directory: str | None = None) -> Session:
        """Get or create session by opencode chat session ID.
        
        Each opencode chat gets its own isolated RLM session.
        This ensures two agents in the same directory don't share context.
        """
        # Check in-memory cache first
        for session in self.sessions.values():
            if session.opencode_session_id == opencode_id:
                return self.get_session(session.id)
        
        # Check database
        with self._get_db() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE opencode_session_id = ?",
                (opencode_id,)
            ).fetchone()
            
            if row:
                session = self._row_to_session(row)
                self.sessions[session.id] = session
                return self.get_session(session.id)
        
        # Create new session bound to this opencode chat
        session = self.create_session(opencode_session_id=opencode_id, directory=directory)
        console.print(f"[green]Created session {session.id} for opencode chat: {opencode_id[:12]}...[/green]")
        return session
    
    def set_incognito(self, session_id: str, incognito: bool):
        """Toggle incognito mode for a session."""
        session = self.get_raw_session(session_id)
        if not session:
            return
        
        if incognito and not session.incognito:
            session.incognito = True
            self._incognito_sessions.add(session_id)
            
            with self._get_db() as conn:
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                conn.execute("DELETE FROM entries WHERE session_id = ?", (session_id,))
            
            context_path = CONTEXTS_DIR / session.context_file
            if context_path.exists():
                with open(context_path) as f:
                    self._incognito_contexts[session_id] = f.read()
                context_path.unlink()
            else:
                self._incognito_contexts[session_id] = ""
            
            console.print(f"[dim]Session {session_id} now in incognito mode[/dim]")
        
        elif not incognito and session.incognito:
            session.incognito = False
            self._incognito_sessions.discard(session_id)
            
            context_data = self._incognito_contexts.pop(session_id, "")
            
            context_path = CONTEXTS_DIR / session.context_file
            with open(context_path, "w") as f:
                f.write(context_data)
            
            self._save_session(session)
            console.print(f"[green]Session {session_id} now persisted[/green]")
    
    def append(
        self,
        session_id: str,
        entry_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContextEntry:
        """Append content to a session."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        with self._locks[session.id]:
            if session.incognito:
                current_size = sum(e.length for e in session.entries)
            else:
                context_path = CONTEXTS_DIR / session.context_file
                current_size = context_path.stat().st_size if context_path.exists() else 0
            
            if session.incognito:
                sep = "\n---ENTRY_SEPARATOR---\n"
                content_with_sep = content + sep if not content.endswith("\n") else content + "---ENTRY_SEPARATOR---\n"
                if session_id not in self._incognito_contexts:
                    self._incognito_contexts[session_id] = ""
                self._incognito_contexts[session_id] += content_with_sep
            else:
                context_path = CONTEXTS_DIR / session.context_file
                with open(context_path, "a") as f:
                    f.write(content)
                    if not content.endswith("\n"):
                        f.write("\n")
                    f.write("---ENTRY_SEPARATOR---\n")
            
            entry = ContextEntry(
                type=entry_type,
                offset=current_size,
                length=len(content),
                timestamp=time.time(),
                metadata=metadata or {},
            )
            
            session.entries.append(entry)
            
            session.stats.total_chars += len(content)
            if entry_type == "file_read":
                session.stats.files_read += 1
            elif entry_type == "tool_result":
                session.stats.tool_outputs += 1
            elif entry_type == "thinking":
                session.stats.thinking_blocks += 1
            
            if not session.incognito:
                self._save_session(session)
            
            return entry
    
    def get_context(self, session_id: str) -> str:
        """Get the full context for a session."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.incognito:
            return self._incognito_contexts.get(session_id, "")
        
        context_path = CONTEXTS_DIR / session.context_file
        if context_path.exists():
            with open(context_path) as f:
                return f.read()
        return ""
    
    def get_context_summary(self, session_id: str) -> str:
        """Get a summary of the session context."""
        session = self.get_session(session_id)
        if not session:
            return ""
        
        stats = session.stats
        if stats.total_chars == 0:
            return ""
        
        total_mb = stats.total_chars / 1_000_000
        mode = " (incognito)" if session.incognito else ""
        
        return f"## RLM Context ({stats.total_chars:,} chars, {total_mb:.2f} MB){mode}\nTools: {stats.tool_outputs} | Files: {stats.files_read} | Thinking: {stats.thinking_blocks}\n"
    
    def forget_context(self, session_id: str, offset: int, length: int, reason: str) -> bool:
        """Permanently redact a section of context and completely rebuild indices."""
        session = self.get_session(session_id)
        if not session:
            return False
            
        with self._locks[session.id]:
            # Read full current context
            if session.incognito:
                full_text = self._incognito_contexts.get(session.id, "")
            else:
                context_path = CONTEXTS_DIR / session.context_file
                if not context_path.exists():
                    return False
                full_text = context_path.read_text(errors="replace")
                
            if offset >= len(full_text) or offset < 0:
                return False
                
            actual_length = min(length, len(full_text) - offset)
                
            tombstone = (
                f"\n===================================================================\n"
                f"[MEMORY REDACTED]\n"
                f"Reason: {reason}\n"
                f"Original size: {actual_length} characters\n"
                f"===================================================================\n"
            )
            
            # Splice
            new_text = full_text[:offset] + tombstone + full_text[offset + actual_length:]
            
            # Wipe current entries and stats
            session.entries = []
            session.stats = SessionStats()
            
            if session.incognito:
                self._incognito_contexts[session.id] = ""
            else:
                context_path.write_text("")
                with self._get_db() as conn:
                    conn.execute("DELETE FROM entries WHERE session_id = ?", (session.id,))
                    
            # Rebuild by appending chunks
            chunks = new_text.split("---ENTRY_SEPARATOR---\n")
            for chunk in chunks:
                if not chunk.strip():
                    continue
                    
                entry_type = "user_message"
                if chunk.startswith("[Tool:"):
                    entry_type = "tool_result"
                elif chunk.startswith("[Thinking]"):
                    entry_type = "thinking"
                elif chunk.startswith("\n====================="):
                    entry_type = "memory_redaction"
                    
                self.append(session.id, entry_type, chunk)
                
        return True
    
    def import_context_file(self, file_path: str) -> Session:
        """Import a raw context.txt file and rebuild the session and indices.
        
        This makes RLM sessions completely portable between machines.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Context file not found: {path}")
            
        content = path.read_text(errors="replace")
        chunks = content.split("---ENTRY_SEPARATOR---\n")
        
        session = self.create_session()
        
        from rich.progress import track
        
        # Parse chunks and append
        for chunk in track(chunks, description="Importing context...", transient=True):
            if not chunk.strip():
                continue
                
            entry_type = "user_message"
            if chunk.startswith("[Tool:"):
                entry_type = "tool_result"
            elif chunk.startswith("[Thinking]"):
                entry_type = "thinking"
                
            # append() automatically handles building DB indices and saving state!
            self.append(session.id, entry_type, chunk)
            
        console.print(f"[green]Successfully imported {len(chunks)} entries to new session [cyan]{session.id}[/cyan][/green]")
        return session

    def branch_session(self, source_session_id: str, drop_last_n: int = 0) -> Session:
        """Create a new session from an existing one, optionally dropping recent entries.
        
        This mimics Git branching for agent memory. If the agent goes down a bad path,
        the user can branch the session from 10 turns ago and attach the new UI to it.
        """
        source = self.get_session(source_session_id)
        if not source:
            raise ValueError(f"Source session {source_session_id} not found")
            
        if source.incognito:
            raise ValueError("Cannot branch an incognito session")
            
        source_path = CONTEXTS_DIR / source.context_file
        if not source_path.exists():
            raise FileNotFoundError(f"Source context file not found: {source_path}")
            
        content = source_path.read_text(errors="replace")
        chunks = content.split("---ENTRY_SEPARATOR---\n")
        
        # Remove empty trailing chunk if present
        if chunks and not chunks[-1].strip():
            chunks.pop()
            
        # Drop recent entries
        if drop_last_n > 0:
            if drop_last_n >= len(chunks):
                raise ValueError(f"Cannot drop {drop_last_n} entries (session only has {len(chunks)})")
            chunks = chunks[:-drop_last_n]
            
        # Create new branch
        session = self.create_session()
        session.parent_session_id = source.id
        self._save_session(session)
        
        from rich.progress import track
        
        for chunk in track(chunks, description=f"Branching session...", transient=True):
            if not chunk.strip():
                continue
                
            entry_type = "user_message"
            if chunk.startswith("[Tool:"):
                entry_type = "tool_result"
            elif chunk.startswith("[Thinking]"):
                entry_type = "thinking"
            elif chunk.startswith("\n====================="):
                entry_type = "memory_redaction"
                
            self.append(session.id, entry_type, chunk)
            
        return session
    
    def build_session_tree(self) -> dict[str, dict]:
        """Reconstruct the hierarchy of session branches.
        
        Returns a dict of root sessions mapped to dictionaries representing their child trees.
        """
        sessions = self.list_sessions(include_incognito=False)
        session_map = {s.id: {"session": s, "children": {}} for s in sessions}
        
        roots = {}
        for s in sessions:
            parent_id = s.parent_session_id
            if parent_id and parent_id in session_map:
                session_map[parent_id]["children"][s.id] = session_map[s.id]
            else:
                # Root branch or orphaned branch whose parent was deleted
                roots[s.id] = session_map[s.id]
                
        return roots
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session = self.get_raw_session(session_id)
        if not session:
            return False
        
        if not session.incognito:
            context_path = CONTEXTS_DIR / session.context_file
            if context_path.exists():
                context_path.unlink()
        
        with self._get_db() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.execute("DELETE FROM entries WHERE session_id = ?", (session_id,))
        
        self.sessions.pop(session_id, None)
        self._incognito_sessions.discard(session_id)
        self._incognito_contexts.pop(session_id, None)
        
        console.print(f"[yellow]Deleted session {session_id}[/yellow]")
        return True
    
    def list_sessions(self, include_incognito: bool = False) -> list[Session]:
        """List all sessions."""
        sessions = []
        
        with self._get_db() as conn:
            for row in conn.execute("SELECT * FROM sessions ORDER BY created DESC"):
                sessions.append(self._row_to_session(row))
        
        if include_incognito:
            for session_id in self._incognito_sessions:
                if session_id in self.sessions:
                    sessions.append(self.sessions[session_id])
        
        return sessions
    
    def sync_with_opencode(self, session_id: str, opencode_session_id: str):
        """Sync an existing session with an opencode session."""
        session = self.get_raw_session(session_id)
        if not session:
            return
        
        session.opencode_session_id = opencode_session_id
        
        if not session.incognito:
            with self._get_db() as conn:
                conn.execute(
                    "UPDATE sessions SET opencode_session_id = ? WHERE id = ?",
                    (opencode_session_id, session_id)
                )
    
    def _save_session(self, session: Session):
        """Save session to database."""
        if session.incognito:
            return
        
        with self._get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (id, created, opencode_session_id, directory, incognito, target_session_id, parent_session_id, context_file, stats_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id,
                session.created,
                session.opencode_session_id,
                session.directory,
                1 if session.incognito else 0,
                session.target_session_id,
                session.parent_session_id,
                session.context_file,
                json.dumps(session.stats.__dict__),
            ))
            
            conn.execute("DELETE FROM entries WHERE session_id = ?", (session.id,))
            
            for entry in session.entries:
                conn.execute("""
                    INSERT INTO entries (session_id, type, offset, length, timestamp, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session.id,
                    entry.type,
                    entry.offset,
                    entry.length,
                    entry.timestamp,
                    json.dumps(entry.metadata),
                ))
    
    def _load_session(self, session_id: str) -> Session | None:
        """Load session from database."""
        with self._get_db() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,)
            ).fetchone()
            
            if not row:
                return None
            
            session = self._row_to_session(row)
            
            for entry_row in conn.execute(
                "SELECT * FROM entries WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            ):
                session.entries.append(ContextEntry(
                    type=entry_row["type"],
                    offset=entry_row["offset"],
                    length=entry_row["length"],
                    timestamp=entry_row["timestamp"],
                    metadata=json.loads(entry_row["metadata_json"] or "{}"),
                ))
            
            return session
    
    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """Convert database row to Session object."""
        stats_data = json.loads(row["stats_json"] or "{}")
        return Session(
            id=row["id"],
            created=row["created"],
            opencode_session_id=row["opencode_session_id"],
            directory=row["directory"],
            incognito=bool(row["incognito"]),
            target_session_id=row["target_session_id"] if "target_session_id" in row.keys() else None,
            parent_session_id=row["parent_session_id"] if "parent_session_id" in row.keys() else None,
            context_file=row["context_file"],
            stats=SessionStats(
                files_read=stats_data.get("files_read", 0),
                tool_outputs=stats_data.get("tool_outputs", 0),
                thinking_blocks=stats_data.get("thinking_blocks", 0),
                total_chars=stats_data.get("total_chars", 0),
            ),
        )


session_manager = SessionManager()
