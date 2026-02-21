"""OpenCode session detection.

Detects the current opencode session from SQLite database.
"""

import sqlite3
import time
from pathlib import Path

from rich.console import Console

console = Console()

OPENCODE_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"


def get_opencode_db_path() -> Path:
    """Get the path to opencode's SQLite database."""
    return OPENCODE_DB


def get_recent_sessions(limit: int = 10) -> list[dict]:
    """Get recent opencode sessions.
    
    Returns list of dicts with:
    - id: opencode session ID
    - slug: session slug
    - title: session title
    - directory: working directory
    - time_created: Unix timestamp
    """
    if not OPENCODE_DB.exists():
        return []
    
    try:
        conn = sqlite3.connect(OPENCODE_DB)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, slug, title, directory, time_created
            FROM session
            ORDER BY time_created DESC
            LIMIT ?
        """, (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "id": row[0],
                "slug": row[1],
                "title": row[2],
                "directory": row[3],
                "time_created": row[4],
            })
        
        conn.close()
        return sessions
        
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to read opencode DB: {e}[/yellow]")
        return []


def get_current_session(cwd: str | None = None) -> dict | None:
    """Get the current opencode session.
    
    Strategy:
    1. Get recent sessions
    2. If cwd provided, prefer session with matching directory
    3. Otherwise, return most recent session
    
    Args:
        cwd: Current working directory (optional)
    
    Returns:
        Session dict or None
    """
    sessions = get_recent_sessions(limit=20)
    
    if not sessions:
        return None
    
    # If cwd provided, try to find matching session
    if cwd:
        cwd_path = Path(cwd).resolve()
        for session in sessions:
            if session.get("directory"):
                try:
                    session_dir = Path(session["directory"]).resolve()
                    if session_dir == cwd_path or cwd_path.is_relative_to(session_dir):
                        return session
                except:
                    pass
    
    # Return most recent
    return sessions[0]


def get_session_by_id(session_id: str) -> dict | None:
    """Get a specific opencode session by ID."""
    if not OPENCODE_DB.exists():
        return None
    
    try:
        conn = sqlite3.connect(OPENCODE_DB)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, slug, title, directory, time_created
            FROM session
            WHERE id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "slug": row[1],
                "title": row[2],
                "directory": row[3],
                "time_created": row[4],
            }
        return None
        
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to read session: {e}[/yellow]")
        return None


def get_messages_for_session(session_id: str, limit: int = 100) -> list[dict]:
    """Get messages for a specific session.
    
    Returns list of message data (JSON strings).
    """
    if not OPENCODE_DB.exists():
        return []
    
    try:
        import json
        conn = sqlite3.connect(OPENCODE_DB)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT data FROM message
            WHERE session_id = ?
            ORDER BY time_created ASC
            LIMIT ?
        """, (session_id, limit))
        
        messages = []
        for row in cursor.fetchall():
            try:
                msg = json.loads(row[0])
                messages.append(msg)
            except:
                pass
        
        conn.close()
        return messages
        
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to read messages: {e}[/yellow]")
        return []


if __name__ == "__main__":
    # Test
    print("Recent sessions:")
    for s in get_recent_sessions(5):
        print(f"  {s['id']}: {s['title'][:40]} ({s['slug']})")
    
    print("\nCurrent session:")
    current = get_current_session()
    if current:
        print(f"  {current['id']}: {current['title']}")
