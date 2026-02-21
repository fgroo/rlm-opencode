"""CLI for RLM-OpenCode Server."""

import os
import signal
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rlm_opencode import __version__

app = typer.Typer(name="rlm-opencode", help="RLM-OpenCode: True RLM for AI coding assistants")
console = Console()

LOG_FILE = "/tmp/rlm-serve.log"
PID_FILE = "/tmp/rlm-opencode.pid"
DEFAULT_PORT = 8769


def _get_server_pid() -> int | None:
    """Get the running server's PID."""
    # Check PID file first
    if Path(PID_FILE).exists():
        try:
            pid = int(Path(PID_FILE).read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            Path(PID_FILE).unlink(missing_ok=True)
    
    # Fallback: check port
    try:
        import subprocess
        result = subprocess.run(
            ["fuser", f"{DEFAULT_PORT}/tcp"],
            capture_output=True, text=True, timeout=2,
        )
        if result.stdout.strip():
            return int(result.stdout.strip().split()[-1])
    except:
        pass
    
    return None


def _is_server_running() -> bool:
    """Check if the server is running."""
    try:
        import httpx
        r = httpx.get(f"http://localhost:{DEFAULT_PORT}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


@app.command()
def serve(
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    background: bool = typer.Option(False, "--bg", "-b", help="Run in background"),
):
    """Start the RLM-OpenCode server (True RLM with tool-based context access)."""
    if _is_server_running():
        console.print(f"[yellow]Server already running on port {port}[/yellow]")
        return
    
    if background:
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, "-m", "rlm_opencode.cli", "serve", "--port", str(port), "--host", host],
            stdout=open(LOG_FILE, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        Path(PID_FILE).write_text(str(proc.pid))
        console.print(f"[green]Server started in background (PID {proc.pid})[/green]")
        console.print(f"[dim]Log: {LOG_FILE}[/dim]")
        return
    
    console.print(f"[bold blue]Starting RLM-OpenCode Server on {host}:{port}[/bold blue]")
    console.print("[dim]True RLM with tool-based context access[/dim]")
    
    # Write PID file for stop/restart
    Path(PID_FILE).write_text(str(os.getpid()))
    
    try:
        import uvicorn
        uvicorn.run("rlm_opencode.server:app", host=host, port=port)
    finally:
        Path(PID_FILE).unlink(missing_ok=True)


@app.command()
def stop():
    """Stop the RLM-OpenCode server."""
    pid = _get_server_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            console.print(f"[green]Server stopped (PID {pid})[/green]")
            Path(PID_FILE).unlink(missing_ok=True)
        except ProcessLookupError:
            console.print("[yellow]Server process not found (already stopped?)[/yellow]")
            Path(PID_FILE).unlink(missing_ok=True)
    else:
        # Fallback: kill by port
        try:
            import subprocess
            subprocess.run(["fuser", "-k", f"{DEFAULT_PORT}/tcp"], capture_output=True, timeout=3)
            console.print(f"[green]Killed process on port {DEFAULT_PORT}[/green]")
        except:
            console.print("[yellow]No server found to stop[/yellow]")


@app.command()
def restart(
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
):
    """Restart the RLM-OpenCode server."""
    console.print("[dim]Stopping server...[/dim]")
    
    pid = _get_server_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        except ProcessLookupError:
            pass
    else:
        # Kill by port
        try:
            import subprocess
            subprocess.run(["fuser", "-k", f"{DEFAULT_PORT}/tcp"], capture_output=True, timeout=3)
            time.sleep(2)
        except:
            pass
    
    Path(PID_FILE).unlink(missing_ok=True)
    
    # Reinstall and start
    console.print("[dim]Reinstalling...[/dim]")
    import subprocess
    subprocess.run(
        ["pip", "install", "-e", ".", "--break-system-packages"],
        capture_output=True, cwd=str(Path(__file__).parent.parent.parent),
    )
    
    console.print(f"[dim]Starting server on {host}:{port}...[/dim]")
    proc = subprocess.Popen(
        [sys.executable, "-m", "rlm_opencode.cli", "serve", "--port", str(port), "--host", host],
        stdout=open(LOG_FILE, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    Path(PID_FILE).write_text(str(proc.pid))
    
    time.sleep(3)
    if _is_server_running():
        console.print(f"[green]✓ Server restarted (PID {proc.pid})[/green]")
    else:
        console.print(f"[red]Server may have failed to start. Check {LOG_FILE}[/red]")


@app.command()
def version():
    """Show version."""
    console.print(f"RLM-OpenCode v{__version__}")


@app.command()
def status():
    """Show server and session status."""
    # Check server
    if _is_server_running():
        pid = _get_server_pid()
        pid_str = f" (PID {pid})" if pid else ""
        console.print(f"[green]● Server RUNNING on port {DEFAULT_PORT}{pid_str}[/green]")
    else:
        console.print(f"[red]● Server NOT RUNNING[/red]")
    
    # Show session count
    from rlm_opencode.session import session_manager
    all_sessions = session_manager.list_sessions()
    console.print(f"[dim]  Sessions: {len(all_sessions)}[/dim]")
    
    # Show env config
    console.print()
    console.print("[dim]Config (env vars):[/dim]")
    env_vars = [
        ("RLM_UPSTREAM_MAX_TOKENS", "128000"),
        ("RLM_TOKEN_RESERVE", "16000"),
        ("RLM_CAPTURE_MIN_CHARS", "500"),
        ("RLM_USER_MIN_CHARS", "0"),
        ("RLM_ASSISTANT_MIN_CHARS", "50"),
        ("RLM_CAPTURE_MAX_CHARS", "50000"),
    ]
    for var, default in env_vars:
        val = os.environ.get(var, default)
        is_custom = os.environ.get(var) is not None
        marker = " [cyan](custom)[/cyan]" if is_custom else ""
        console.print(f"  {var}={val}{marker}")


@app.command()
def sessions():
    """List all RLM sessions with context stats."""
    from rlm_opencode.session import session_manager
    
    all_sessions = session_manager.list_sessions()
    if not all_sessions:
        console.print("[dim]No sessions found.[/dim]")
        return
    
    table = Table(title="RLM Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Fingerprint", style="dim")
    table.add_column("Created", style="dim")
    table.add_column("Context", justify="right", style="green")
    table.add_column("Entries", justify="right")
    table.add_column("Context File", style="dim")
    
    for session in all_sessions:
        fingerprint = session.opencode_session_id or "—"
        created = time.strftime("%m/%d %H:%M", time.localtime(session.created))
        
        # Get context size
        context_chars = session.stats.total_chars if session.stats else 0
        if context_chars > 1_000_000:
            context_str = f"{context_chars / 1_000_000:.1f}M"
        elif context_chars > 1_000:
            context_str = f"{context_chars / 1_000:.1f}K"
        else:
            context_str = str(context_chars)
        
        entry_count = len(session.entries) if session.entries else 0
        
        # Context file path (shortened)
        ctx_file = session.context_file
        if ctx_file:
            ctx_file = str(ctx_file).replace(str(Path.home()), "~")
        else:
            ctx_file = "—"
        
        table.add_row(
            session.id,
            fingerprint[:12] + "..." if len(fingerprint) > 12 else fingerprint,
            created,
            context_str,
            str(entry_count),
            ctx_file,
        )
    
    console.print(table)


@app.command()
def log(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output (tail -f)"),
    full: bool = typer.Option(False, "--full", help="Show entire log"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
):
    """View RLM-OpenCode server log.
    
    Examples:
        rlm-opencode log              # Last 50 lines
        rlm-opencode log -f           # Follow mode (live updates)
        rlm-opencode log --full       # Entire log
        rlm-opencode log -n 100       # Last 100 lines
    """
    if not os.path.exists(LOG_FILE):
        console.print(f"[yellow]Log file not found: {LOG_FILE}[/yellow]")
        console.print("[dim]Start the server first: rlm-opencode serve[/dim]")
        return
    
    if follow:
        console.print(f"[dim]Following {LOG_FILE} (Ctrl+C to stop)...[/dim]")
        os.system(f"tail -f {LOG_FILE}")
    elif full:
        with open(LOG_FILE) as f:
            console.print(f.read())
    else:
        with open(LOG_FILE) as f:
            all_lines = f.readlines()[-lines:]
            console.print("".join(all_lines))


@app.command()
def models(
    provider: str = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    search: str = typer.Option(None, "--search", "-s", help="Search models by name"),
):
    """List available models from opencode config."""
    from rlm_opencode.providers.registry import get_registry
    
    registry = get_registry()
    
    console.print(f"[dim]Providers: {', '.join(registry.list_providers())}[/dim]")
    console.print(f"[dim]Total model mappings: {registry.count_models()}[/dim]")
    console.print()
    
    table = Table(title="Available Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Provider", style="dim")
    
    seen = set()
    for model_id, provider_id in registry.model_to_provider.items():
        if model_id in seen:
            continue
        if "/" not in model_id:
            continue
        seen.add(model_id)
        
        if provider and provider not in provider_id:
            continue
        if search and search.lower() not in model_id.lower():
            continue
        
        table.add_row(model_id, provider_id)
    
    console.print(table)


@app.command()
def clear(
    session_data: bool = typer.Option(False, "--sessions", help="Clear all sessions"),
    log_file: bool = typer.Option(False, "--log", help="Clear log file"),
    all_: bool = typer.Option(False, "--all", "-a", help="Clear everything"),
):
    """Clear sessions or log files."""
    if all_:
        session_data = True
        log_file = True
    
    if session_data:
        from rlm_opencode.session import RLM_DATA_DIR
        import shutil
        if RLM_DATA_DIR.exists():
            shutil.rmtree(RLM_DATA_DIR)
            console.print("[green]Cleared all sessions and data[/green]")
    
    if log_file:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            console.print(f"[green]Cleared {LOG_FILE}[/green]")


def main():
    app()


if __name__ == "__main__":
    main()
