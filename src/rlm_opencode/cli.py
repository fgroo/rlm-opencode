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
            [sys.executable, "-u", "-m", "rlm_opencode.cli", "serve", "--port", str(port), "--host", host],
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
        [sys.executable, "-u", "-m", "rlm_opencode.cli", "serve", "--port", str(port), "--host", host],
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
        ("RLM_MAX_PAYLOAD_CHARS", "250000"),
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

    cfg = session_manager.get_config()
    console.print()
    console.print("[dim]Config (Persistent json config):[/dim]")
    if not cfg:
        console.print("  [dim]No persistent config set.[/dim]")
    for k, v in cfg.items():
        console.print(f"  {k} = [cyan]{v}[/cyan]")


@app.command()
def strict(
    level: int = typer.Argument(None, help="Strict mode level (0-4)"),
):
    """View or set the Strict Mode level (0 = Off, 4 = Maximum).
    
    Strict Mode forces the LLM to use tools instead of guessing when
    history is truncated. Standard is 0.
    """
    from rlm_opencode.session import session_manager
    
    if level is None:
        current = session_manager.get_config().get("strict_mode_level", 0)
        console.print(f"Current Strict Mode level: [cyan]{current}[/cyan]")
        console.print("[dim]Use 'rlm-opencode strict <0-4>' to change it.[/dim]")
        return
        
    if not 0 <= level <= 4:
        console.print("[red]Error:[/red] Strict mode level must be between 0 and 4.")
        raise typer.Exit(1)
        
    session_manager.set_strict_mode(level)
    
    colors = {
        0: "dim",
        1: "blue",
        2: "magenta",
        3: "red",
        4: "bold red blink"
    }
    style = colors[level]
    
    console.print(f"Strict Mode updated to: [{style}]Level {level}[/{style}]")
    if level > 0:
        console.print("The RLM server will now aggressively instruct the model to use context tools.")


@app.command()
def config(
    key: str = typer.Argument(None, help="Config key to set or view"),
    value: str = typer.Argument(None, help="Value to set"),
):
    """View or set persistent configuration variables (e.g. rlm_max_payload_chars)."""
    from rlm_opencode.session import session_manager
    from rlm_opencode.server import RLM_DEFAULT_SETTINGS, RLM_SETTING_DESCRIPTIONS, get_setting
    
    cfg = session_manager.get_config()
    
    if key is None:
        console.print("[bold]Current Active Configuration:[/bold]")
        console.print("[dim](Priority: JSON > ENV > Default)[/dim]\n")
        
        # Always show strict mode
        strict_val = cfg.get("strict_mode_level", 0)
        source = "[yellow](custom JSON)[/yellow]" if "strict_mode_level" in cfg else "[dim](default)[/dim]"
        desc = RLM_SETTING_DESCRIPTIONS.get("strict_mode_level", "")
        console.print(f"  [bold]strict_mode_level[/bold]: [cyan]{strict_val}[/cyan] {source}")
        console.print(f"    [dim italic]↳ {desc}[/dim italic]")
        
        # Iterating over all known default keys to build a comprehensive list
        for default_key in RLM_DEFAULT_SETTINGS.keys():
            val = get_setting(default_key)
            if default_key in cfg:
                source = "[yellow](custom JSON)[/yellow]"
            elif default_key.upper() in os.environ:
                source = "[blue](from ENV)[/blue]"
            else:
                source = "[dim](default)[/dim]"
            
            desc = RLM_SETTING_DESCRIPTIONS.get(default_key, "")
            console.print(f"\n  [bold]{default_key}[/bold]: [cyan]{val}[/cyan] {source}")
            console.print(f"    [dim italic]↳ {desc}[/dim italic]")
            
        # Check for unrecognized keys
        unknown_keys = [k for k in cfg.keys() if k not in RLM_DEFAULT_SETTINGS and k != "strict_mode_level"]
        if unknown_keys:
            console.print("\n[yellow]Unrecognized Custom Keys:[/yellow]")
            for k in unknown_keys:
                console.print(f"  [bold]{k}[/bold]: [red]{cfg[k]}[/red] [dim](unknown)[/dim]")
            
        console.print()
        return
        
    if value is None:
        if key in cfg:
            console.print(f"{key}: [cyan]{cfg[key]}[/cyan]")
        else:
            console.print(f"[dim]Key '{key}' not set in config.[/dim]")
        return
        
    if value.lower() == "default":
        if key in cfg:
            cfg.pop(key)
            session_manager._save_config(cfg)
            console.print(f"[green]Restored to default:[/green] {key}")
            console.print("[dim]Restart the server for changes to take full effect.[/dim]")
        else:
            console.print(f"[dim]'{key}' is already at its default/env value.[/dim]")
        return
        
    if key != "rlm_summarize_model":
        try:
            parsed_val = int(value)
            if parsed_val < 0:
                console.print(f"[red]Error:[/red] '{key}' must be a positive integer.")
                raise typer.Exit(1)
        except ValueError:
            console.print(f"[red]Error:[/red] '{key}' must be a positive integer.")
            raise typer.Exit(1)
            
        if key == "strict_mode_level" and not (0 <= parsed_val <= 4):
            console.print(f"[red]Error:[/red] '{key}' must be between 0 and 4.")
            raise typer.Exit(1)
    else:
        # rlm_summarize_model can be a string, or None (if passed "None", maybe clear it, but "default" handles clearing)
        parsed_val = value if value.lower() != "none" else None
            
    if key not in RLM_DEFAULT_SETTINGS and key != "strict_mode_level":
        console.print(f"[yellow]Warning: '{key}' is not a recognized configuration setting.[/yellow]")
            
    cfg[key] = parsed_val
    session_manager._save_config(cfg)
    console.print(f"[green]Set config:[/green] {key} = {parsed_val}")
    console.print("[dim]Restart the server for changes to take full effect.[/dim]")


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
        
        # Context file path (shortened) or linked status
        if session.target_session_id:
            ctx_file = f"→ [cyan]{session.target_session_id}[/cyan]"
        else:
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
def join(
    target_session: str = typer.Argument(..., help="The ID of the master session with the context"),
    session_to_redirect: str = typer.Argument(..., help="The ID of the session that will be redirected"),
):
    """Link a session to another session's context (Party Mode).
    
    Like dup2(), this redirects session_to_redirect's reads/writes to target_session.
    Both agents will now share the exact same context!
    """
    from rlm_opencode.session import session_manager
    
    if session_manager.set_target_session(session_to_redirect, target_session):
        console.print(f"[green]🎉 Party Mode Activated![/green]")
        console.print(f"Session [cyan]{session_to_redirect}[/cyan] is now linked to [cyan]{target_session}[/cyan]")
    else:
        console.print(f"[red]Error:[/red] Could not find session {session_to_redirect}")


@app.command(name="import")
def import_ctx(
    file_path: str = typer.Argument(..., help="Path to the raw context.txt file to import"),
):
    """Import a raw context.txt file from another machine into a new session."""
    from rlm_opencode.session import session_manager
    
    try:
        session = session_manager.import_context_file(file_path)
        console.print(f"\n[bold green]Import Complete![/bold green]")
        console.print(f"You can now link your active OpenCode chat to this imported session:")
    except Exception as e:
        console.print(f"[red]Error importing context:[/red] {e}")


@app.command()
def branch(
    source_session_id: str = typer.Argument(..., help="The origin session ID to branch from"),
    drop_last: int = typer.Option(0, "--drop-last", "-d", help="Number of recent context entries to permanently drop in the new branch"),
):
    """Clone an existing session, optionally dropping recent mistakes from memory.
    
    This is like Git for Agent Memory. If the agent goes down a huge rabbit hole
    of failed debugging, you can branch the session from 10 turns ago, leaving
    the garbage behind, and attach your active OpenCode chat to the clean branch.
    """
    from rlm_opencode.session import session_manager
    
    try:
        new_session = session_manager.branch_session(source_session_id, drop_last)
        console.print(f"\n[bold green]Branch Created Successfully![/bold green]")
        console.print(f"Origin: [dim]{source_session_id}[/dim]")
        if drop_last > 0:
            console.print(f"Dropped: [red]last {drop_last} entries[/red]")
            
        console.print(f"\n🚀 New Clean Branch: [cyan]{new_session.id}[/cyan]")
        console.print(f"To use this branch in your active OpenCode window, run:")
        console.print(f"  [cyan]rlm-opencode join <your_active_chat_id> {new_session.id}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error branching session:[/red] {str(e)}")


@app.command()
def unjoin(
    session_id: str = typer.Argument(..., help="The ID of the session to un-link"),
):
    """Remove a session link, restoring its private context."""
    from rlm_opencode.session import session_manager
    
    if session_manager.set_target_session(session_id, None):
        console.print(f"[yellow]Link removed.[/yellow]")
        console.print(f"Session [cyan]{session_id}[/cyan] now has its own private context again.")
    else:
        console.print(f"[red]Error:[/red] Could not find session {session_id}")


@app.command()
def server_logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output (tail -f)"),
    full: bool = typer.Option(False, "--full", help="Show entire log"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
):
    """View RLM-OpenCode server log.
    
    Examples:
        rlm-opencode server-logs              # Last 50 lines
        rlm-opencode server-logs -f           # Follow mode (live updates)
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
def log():
    """View the visual session tree (Git for Agent Memory).
    
    Displays a hierarchical representation of all sessions and their branches.
    """
    from rlm_opencode.session import session_manager
    from rich.tree import Tree
    
    roots = session_manager.build_session_tree()
    
    if not roots:
        console.print("[dim]No sessions found in the memory lake.[/dim]")
        return
        
    def _format_node(session) -> str:
        # Determine color and activity
        color = "cyan" if session.opencode_session_id else "dim"
        active_tag = " [bold green]★ ACTIVE[/bold green]" if session.opencode_session_id else ""
        
        # Format timestamps
        created = time.strftime("%b %d %H:%M", time.localtime(session.created))
        
        # Formatting context size
        chars = session.stats.total_chars if session.stats else 0
        if chars > 1_000_000:
            size_str = f"{chars / 1_000_000:.1f}MB"
        elif chars > 1_000:
            size_str = f"{chars / 1_000:.1f}KB"
        else:
            size_str = f"{chars}B"
            
        entries = len(session.entries) if session.entries else 0
        
        return f"[{color}]{session.id}[/{color}] [dim]• {created} • {size_str} ({entries} turns)[/dim]{active_tag}"
        
    def _build_tree(node_dict: dict, tree: Tree):
        # Sort children by creation time
        children = dict(sorted(node_dict["children"].items(), key=lambda x: x[1]["session"].created))
        
        for child_id, child_node in children.items():
            child_branch = tree.add(_format_node(child_node["session"]))
            _build_tree(child_node, child_branch)
            
    # Main Forest rendering
    console.print("\n[bold]🌳 RLM Session Memory Tree[/bold]\n")
    
    # Sort roots by creation time (newest last)
    sorted_roots = sorted(roots.values(), key=lambda x: x["session"].created)
    
    for root_node in sorted_roots:
        root_tree = Tree(_format_node(root_node["session"]))
        _build_tree(root_node, root_tree)
        console.print(root_tree)
        console.print()


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
