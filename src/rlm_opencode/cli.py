"""CLI for RLM-OpenCode Server."""

import os
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rlm_opencode import __version__

app = typer.Typer(name="rlm-opencode", help="RLM-OpenCode: True RLM for AI coding assistants")
console = Console()

LOG_FILE = "/tmp/rlm-opencode.log"
SESSIONS_DIR = Path.home() / ".local" / "share" / "rlm-opencode" / "sessions"
MAPPINGS_DIR = Path.home() / ".local" / "share" / "rlm-opencode" / "mappings"


@app.command()
def serve(
    port: int = typer.Option(8769, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
):
    """Start the RLM-OpenCode server (True RLM with tool-based context access)."""
    console.print(f"[bold blue]Starting RLM-OpenCode Server on {host}:{port}[/bold blue]")
    console.print("[dim]True RLM with tool-based context access[/dim]")
    import uvicorn
    uvicorn.run(
        "rlm_opencode.server:app",
        host=host,
        port=port,
    )


@app.command()
def version():
    """Show version."""
    console.print(f"RLM Session v{__version__}")


@app.command()
def status():
    """Show server and session status."""
    import httpx
    import json
    
    # Check multiple servers
    servers = [
        ("RLM-Server", 8765),
        ("RLM-Session (Proxy)", 8767),
        ("RLM-Session (Native)", 8768),
    ]
    
    console.print("[bold]Servers:[/bold]")
    for name, port in servers:
        try:
            response = httpx.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                mode = data.get("mode", "proxy")
                console.print(f"  [green]{name}[/green]: RUNNING (port {port}, {mode})")
            else:
                console.print(f"  [red]{name}[/red]: ERROR (port {port})")
        except:
            console.print(f"  [dim]{name}[/dim]: NOT RUNNING (port {port})")
    
    console.print()
    
    # Directory mapping status
    dir_mapping_file = MAPPINGS_DIR / "directory_to_rlm.json"
    if dir_mapping_file.exists():
        with open(dir_mapping_file) as f:
            mappings = json.load(f)
        
        if mappings:
            console.print("[cyan]Active Sessions:[/cyan]")
            for directory, info in mappings.items():
                session_id = info["rlm_opencode_id"]
                session_file = SESSIONS_DIR / f"{session_id}.json"
                
                if session_file.exists():
                    with open(session_file) as f:
                        session_data = json.load(f)
                    
                    stats = session_data.get("stats", {})
                    created = time.strftime("%Y-%m-%d %H:%M", time.localtime(info["created"]))
                    
                    console.print(f"  [bold]{session_id}[/bold]")
                    console.print(f"    Directory: {directory}")
                    console.print(f"    Created: {created}")
                    console.print(f"    Context: {stats.get('total_chars', 0):,} chars")
                    console.print(f"    Files: {stats.get('files_read', 0)}, Commands: {stats.get('commands_run', 0)}")
                    console.print()
        else:
            console.print("[dim]No active sessions[/dim]")
    else:
        console.print("[dim]No sessions created yet[/dim]")


@app.command()
def log(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output (tail -f)"),
    full: bool = typer.Option(False, "--full", help="Show entire log"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Show raw log with ANSI codes"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    native: bool = typer.Option(False, "--native", "-N", help="View native server log (default: proxy)"),
):
    """View RLM-Session activity log.
    
    Examples:
        rlm-opencode log              # Last 50 lines of proxy log (parsed)
        rlm-opencode log -f           # Follow mode (live updates)
        rlm-opencode log --full       # Entire log
        rlm-opencode log --raw        # Raw output with ANSI codes
        rlm-opencode log -n 100       # Last 100 lines
        rlm-opencode log --native     # View native server log
    """
    log_file = LOG_FILE_NATIVE if native else LOG_FILE_PROXY
    
    if not os.path.exists(log_file):
        mode = "native" if native else "proxy"
        console.print(f"[yellow]Log file not found. {mode.title()} server may not have been started yet.[/yellow]")
        console.print(f"Expected location: {log_file}")
        return
    
    if follow:
        console.print(f"[dim]Following {log_file} (Ctrl+C to stop)...[/dim]")
        os.system(f"tail -f {log_file}")
    elif raw:
        if full:
            os.system(f"cat {log_file}")
        else:
            os.system(f"tail -{lines} {log_file}")
    else:
        # Parsed/clean output
        if full:
            with open(log_file) as f:
                content = f.read()
        else:
            with open(log_file) as f:
                lines_list = f.readlines()[-lines:]
                content = "".join(lines_list)
        
        # Parse and clean up
        _print_parsed_log(content)


@app.command()
def sessions():
    """List all RLM sessions with stats."""
    import json
    
    if not SESSIONS_DIR.exists():
        console.print("[dim]No sessions directory found.[/dim]")
        return
    
    session_files = list(SESSIONS_DIR.glob("*.json"))
    if not session_files:
        console.print("[dim]No sessions found.[/dim]")
        return
    
    table = Table(title="RLM Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Context", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Cmds", justify="right")
    table.add_column("Msgs", justify="right")
    
    for session_file in sorted(session_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(session_file) as f:
                data = json.load(f)
            
            session_id = data.get("id", "unknown")
            created = time.strftime("%m/%d %H:%M", time.localtime(data.get("created", 0)))
            stats = data.get("stats", {})
            
            context_chars = stats.get("total_chars", 0)
            if context_chars > 1_000_000:
                context_str = f"{context_chars / 1_000_000:.1f}M"
            elif context_chars > 1_000:
                context_str = f"{context_chars / 1_000:.1f}K"
            else:
                context_str = str(context_chars)
            
            table.add_row(
                session_id,
                created,
                context_str,
                str(stats.get("files_read", 0)),
                str(stats.get("commands_run", 0)),
                str(stats.get("user_messages", 0)),
            )
        except Exception as e:
            console.print(f"[red]Error reading {session_file}: {e}[/red]")
    
    console.print(table)
    
    # Also show directory mappings
    dir_mapping_file = MAPPINGS_DIR / "directory_to_rlm.json"
    if dir_mapping_file.exists():
        with open(dir_mapping_file) as f:
            mappings = json.load(f)
        
        if mappings:
            console.print()
            console.print("[dim]Directory Mappings:[/dim]")
            for directory, info in mappings.items():
                console.print(f"  {info['rlm_opencode_id']}: {directory}")


@app.command()
def models(
    provider: str = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    search: str = typer.Option(None, "--search", "-s", help="Search models by name"),
):
    """List available models from opencode.
    
    Examples:
        rlm-opencode models
        rlm-opencode models -p openai
        rlm-opencode models -s gpt
    """
    from rlm_opencode.providers.registry import get_registry
    
    registry = get_registry()
    
    console.print(f"[dim]Providers: {', '.join(registry.list_providers())}[/dim]")
    console.print(f"[dim]Total model mappings: {registry.count_models()}[/dim]")
    console.print()
    
    # Show provider/model table
    table = Table(title="Available Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Provider", style="dim")
    
    # Get unique models (avoid duplicates)
    seen = set()
    for model_id, provider_id in registry.model_to_provider.items():
        if model_id in seen:
            continue
        if "/" not in model_id:  # Only show full model IDs
            continue
        seen.add(model_id)
        
        # Filter by provider
        if provider and provider not in provider_id:
            continue
        
        # Filter by search
        if search and search.lower() not in model_id.lower():
            continue
        
        table.add_row(model_id, provider_id)
    
    console.print(table)


@app.command()
def clear(
    sessions: bool = typer.Option(False, "--sessions", help="Clear all sessions"),
    log_file: bool = typer.Option(False, "--log", help="Clear log files"),
    all_: bool = typer.Option(False, "--all", "-a", help="Clear everything"),
):
    """Clear sessions or log files."""
    if all_:
        sessions = True
        log_file = True
    
    if sessions:
        if SESSIONS_DIR.exists():
            import shutil
            shutil.rmtree(SESSIONS_DIR)
            console.print("[green]Cleared all sessions[/green]")
    
    if log_file:
        cleared = []
        for log_path in [LOG_FILE_PROXY, LOG_FILE_NATIVE]:
            if os.path.exists(log_path):
                os.remove(log_path)
                cleared.append(log_path)
        if cleared:
            console.print(f"[green]Cleared log files: {', '.join(cleared)}[/green]")


def _print_parsed_log(content: str):
    """Print log with cleaned formatting."""
    import re
    
    # Remove ANSI escape sequences
    ansi_pattern = r'\x1b\[[0-9;]*m'
    
    for line in content.split("\n"):
        # Remove ANSI codes
        clean = re.sub(ansi_pattern, '', line)
        
        # Colorize based on content
        if "[API]" in clean:
            if "Error" in clean:
                console.print(f"[red]{clean}[/red]")
            elif "Response ready" in clean:
                console.print(f"[green]{clean}[/green]")
            else:
                console.print(f"[cyan]{clean}[/cyan]")
        elif "[LLM]" in clean:
            if "Timeout" in clean or "Error" in clean:
                console.print(f"[red]{clean}[/red]")
            elif "Response in" in clean:
                console.print(f"[green]{clean}[/green]")
            else:
                console.print(f"[yellow]{clean}[/yellow]")
        elif "[CODE]" in clean:
            console.print(f"[magenta]{clean}[/magenta]")
        elif "Created session" in clean or "directory-mapped" in clean:
            console.print(f"[blue]{clean}[/blue]")
        elif "DEBUG" in clean:
            console.print(f"[dim]{clean}[/dim]")
        elif clean.strip():
            console.print(clean)


def main():
    app()


if __name__ == "__main__":
    main()
