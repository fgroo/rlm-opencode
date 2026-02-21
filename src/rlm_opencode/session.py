"""Session management for RLM Session Server.

Handles:
- Session creation and lifecycle
- Context accumulation (file reads, commands, etc.)
- Session persistence and restoration
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

RLM_DATA_DIR = Path.home() / ".local" / "share" / "rlm-opencode"
SESSIONS_DIR = RLM_DATA_DIR / "sessions"
MAPPINGS_DIR = RLM_DATA_DIR / "mappings"


@dataclass
class ContextEntry:
    """A single entry in the session context."""
    type: str  # file_read, command, user_message, thinking
    offset: int
    length: int
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionStats:
    """Statistics about a session."""
    files_read: int = 0
    commands_run: int = 0
    user_messages: int = 0
    thinking_blocks: int = 0
    total_chars: int = 0


@dataclass
class Session:
    """Represents an RLM session."""
    id: str
    created: float
    opencode_session_id: str | None = None
    entries: list[ContextEntry] = field(default_factory=list)
    stats: SessionStats = field(default_factory=SessionStats)
    context_file: str = ""
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created": self.created,
            "opencode_session_id": self.opencode_session_id,
            "context_file": self.context_file,
            "stats": {
                "files_read": self.stats.files_read,
                "commands_run": self.stats.commands_run,
                "user_messages": self.stats.user_messages,
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
            files_read=data["stats"].get("files_read", 0),
            commands_run=data["stats"].get("commands_run", 0),
            user_messages=data["stats"].get("user_messages", 0),
            thinking_blocks=data["stats"].get("thinking_blocks", 0),
            total_chars=data["stats"].get("total_chars", 0),
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
            context_file=data.get("context_file", ""),
            entries=entries,
            stats=stats,
        )


class SessionManager:
    """Manages RLM sessions.
    
    Features:
    - Create/destroy sessions
    - Append context entries
    - Persist and restore sessions
    - Map opencode sessions to RLM sessions
    """
    
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Ensure data directories exist."""
        RLM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    def create_session(self, opencode_session_id: str | None = None) -> Session:
        """Create a new session."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        context_file = f"{session_id}_context.txt"
        
        session = Session(
            id=session_id,
            created=time.time(),
            opencode_session_id=opencode_session_id,
            context_file=context_file,
        )
        
        # Create empty context file
        context_path = SESSIONS_DIR / context_file
        context_path.touch()
        
        self.sessions[session_id] = session
        
        # Save mapping if opencode session provided
        if opencode_session_id:
            self._save_mapping(opencode_session_id, session_id)
        
        console.print(f"[green]Created session {session_id}[/green]")
        return session
    
    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Try to load from disk
        session = self._load_session(session_id)
        if session:
            self.sessions[session_id] = session
        return session
    
    def get_session_by_opencode(self, opencode_session_id: str) -> Session | None:
        """Get session by opencode session ID."""
        mapping = self._load_mapping(opencode_session_id)
        if mapping:
            return self.get_session(mapping["rlm_opencode_id"])
        return None
    
    def get_or_create_session_by_directory(self, directory: str) -> Session:
        """Get or create session by working directory.
        
        This provides stable session mapping across multiple opencode run calls
        from the same project directory.
        """
        # Normalize directory path
        dir_path = str(Path(directory).resolve())
        
        # Check for existing mapping
        mapping = self._load_directory_mapping(dir_path)
        if mapping:
            session = self.get_session(mapping["rlm_opencode_id"])
            if session:
                return session
        
        # Create new session with directory mapping
        session = self.create_session()
        self._save_directory_mapping(dir_path, session.id)
        console.print(f"[green]Created session {session.id} for directory: {dir_path}[/green]")
        return session
    
    def _save_directory_mapping(self, directory: str, rlm_opencode_id: str):
        """Save directory → RLM session mapping."""
        mapping_file = MAPPINGS_DIR / "directory_to_rlm.json"
        
        mappings = {}
        if mapping_file.exists():
            with open(mapping_file) as f:
                mappings = json.load(f)
        
        mappings[directory] = {
            "rlm_opencode_id": rlm_opencode_id,
            "created": time.time(),
        }
        
        with open(mapping_file, "w") as f:
            json.dump(mappings, f, indent=2)
    
    def _load_directory_mapping(self, directory: str) -> dict | None:
        """Load directory → RLM session mapping."""
        mapping_file = MAPPINGS_DIR / "directory_to_rlm.json"
        
        if not mapping_file.exists():
            return None
        
        with open(mapping_file) as f:
            mappings = json.load(f)
        
        return mappings.get(directory)
    
    def append(
        self,
        session_id: str,
        entry_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContextEntry:
        """Append content to a session.
        
        Args:
            session_id: Session ID
            entry_type: Type of entry (file_read, command, user_message, thinking)
            content: The content to append
            metadata: Optional metadata (path, command, etc.)
        
        Returns:
            The created ContextEntry
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        context_path = SESSIONS_DIR / session.context_file
        
        # Get current offset
        current_size = context_path.stat().st_size
        
        # Append content
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
        
        # Update stats
        session.stats.total_chars += len(content)
        if entry_type == "file_read":
            session.stats.files_read += 1
        elif entry_type == "command":
            session.stats.commands_run += 1
        elif entry_type == "user_message":
            session.stats.user_messages += 1
        elif entry_type == "thinking":
            session.stats.thinking_blocks += 1
        
        # Save session metadata
        self._save_session(session)
        
        return entry
    
    def get_context(self, session_id: str) -> str:
        """Get the full context for a session."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        context_path = SESSIONS_DIR / session.context_file
        with open(context_path) as f:
            return f.read()
    
    def get_context_summary(self, session_id: str) -> str:
        """Get a summary of the session context for the LLM prompt."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        stats = session.stats
        total_mb = stats.total_chars / 1_000_000
        
        if stats.total_chars == 0:
            return ""
        
        summary = f"""## RLM Accumulated Context ({stats.total_chars:,} chars, {total_mb:.1f} MB)
Files read: {stats.files_read} | Tool outputs: {stats.commands_run} | Thinking blocks: {stats.thinking_blocks}

"""
        return summary
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Delete context file
        context_path = SESSIONS_DIR / session.context_file
        if context_path.exists():
            context_path.unlink()
        
        # Delete session metadata
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            meta_path.unlink()
        
        # Remove from memory
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        console.print(f"[yellow]Deleted session {session_id}[/yellow]")
        return True
    
    def _save_session(self, session: Session):
        """Save session metadata to disk."""
        meta_path = SESSIONS_DIR / f"{session.id}.json"
        with open(meta_path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)
    
    def _load_session(self, session_id: str) -> Session | None:
        """Load session metadata from disk."""
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if not meta_path.exists():
            return None
        
        with open(meta_path) as f:
            data = json.load(f)
        
        return Session.from_dict(data)
    
    def _save_mapping(self, opencode_session_id: str, rlm_opencode_id: str):
        """Save opencode → RLM session mapping."""
        mapping_file = MAPPINGS_DIR / "opencode_to_rlm.json"
        
        mappings = {}
        if mapping_file.exists():
            with open(mapping_file) as f:
                mappings = json.load(f)
        
        mappings[opencode_session_id] = {
            "rlm_opencode_id": rlm_opencode_id,
            "created": time.time(),
        }
        
        with open(mapping_file, "w") as f:
            json.dump(mappings, f, indent=2)
    
    def _load_mapping(self, opencode_session_id: str) -> dict | None:
        """Load opencode → RLM session mapping."""
        mapping_file = MAPPINGS_DIR / "opencode_to_rlm.json"
        
        if not mapping_file.exists():
            return None
        
        with open(mapping_file) as f:
            mappings = json.load(f)
        
        return mappings.get(opencode_session_id)


# Global session manager
session_manager = SessionManager()
