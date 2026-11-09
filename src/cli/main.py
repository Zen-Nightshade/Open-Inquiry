from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from repo_paths import get_db_path
from src.config.settings import init_env
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration import (
    ApplicationOrchestrator,
    RunResult,
    RunStatus,
    StageExecutionRecord,
)

# Initialize console for Rich terminal rendering
console = Console()

app = typer.Typer(
    name="zamp",
    help="Zamp: Autonomous Evidence-Grounded Research & Outreach Engine",
    add_completion=False,
)

db_app = typer.Typer(help="Manage and inspect SQLite database and schema.")
run_app = typer.Typer(no_args_is_help=False, help="Execute and inspect application runs.")
research_app = typer.Typer(help="Inspect research sessions, canonical facts, and evidence.")
strategy_app = typer.Typer(help="Inspect strategy blueprints and domain explainability.")

app.add_typer(db_app, name="db")
app.add_typer(run_app, name="run")
app.add_typer(research_app, name="research")
app.add_typer(strategy_app, name="strategy")


def _get_orchestrator(db_path: Optional[str] = None) -> ApplicationOrchestrator:
    init_env()
    store = StateStore(db_path=db_path)
    return ApplicationOrchestrator(state_store=store)


# =============================================================================
# Database Commands
# =============================================================================

@db_app.command("init")
def db_init(
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Custom SQLite database path"),
):
    """Initialize database schema directly from schema.sql."""
    init_env()
    store = StateStore(db_path=db_path)
    console.print(f"[cyan]Initializing database schema at:[/cyan] {store.db_path}")

    store.initialize()
    status = store.get_database_status()

    if status["is_schema_valid"]:
        console.print(
            f"[bold green]Database schema initialized successfully. "
            f"All {status['total_expected_tables']} domain tables verified.[/bold green]"
        )
    else:
        console.print(f"[bold red]Database schema verification failed: {status['incompatibility_details']}[/bold red]")
        raise typer.Exit(1)


@db_app.command("status")
def db_status(
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Custom SQLite database path"),
):
    """Inspect database state, existence, and schema validity."""
    init_env()
    store = StateStore(db_path=db_path)
    status = store.get_database_status()

    grid = Table.grid(padding=1)
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column(style="white")

    grid.add_row("Database Path:", str(status["db_path"]))
    grid.add_row("File Exists:", "[green]Yes[/green]" if status["file_exists"] else "[red]No[/red]")
    grid.add_row("Initialized:", "[green]Yes[/green]" if status["is_initialized"] else "[red]No[/red]")
    grid.add_row("Schema Valid:", "[green]Yes[/green]" if status["is_schema_valid"] else "[red]No[/red]")
    grid.add_row("Expected Tables Found:", f"{status['expected_tables_found']} / {status['total_expected_tables']}")
    grid.add_row("Total Tables in DB:", str(status["table_count"]))

    border_color = "green" if status["is_schema_valid"] else ("yellow" if status["file_exists"] else "red")
    console.print(Panel(grid, title="[bold]Database Status[/bold]", border_style=border_color))

    if status["missing_tables"] and status["file_exists"]:
        console.print(f"[bold yellow]Missing Expected Tables ({len(status['missing_tables'])}):[/bold yellow] {', '.join(status['missing_tables'])}")


# =============================================================================
# Run Commands
# =============================================================================

@run_app.callback(invoke_without_command=True)
def run_pipeline(
    ctx: typer.Context,
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Prospect full name"),
    company: Optional[str] = typer.Option(None, "--company", "-c", help="Target company name"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Company domain"),
    designation: Optional[str] = typer.Option(None, "--designation", "-r", help="Prospect role or title"),
    linkedin: Optional[str] = typer.Option(None, "--linkedin", "-l", help="Prospect LinkedIn URL"),
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-f", help="JSON file containing prospect, product, outreach contexts"),
    product_name: Optional[str] = typer.Option(None, "--product", "-p", help="Product name"),
    product_desc: Optional[str] = typer.Option(None, "--product-desc", help="Product description"),
    value_prop: Optional[str] = typer.Option(None, "--value-prop", help="Product value proposition"),
    capabilities: Optional[str] = typer.Option(None, "--capabilities", help="Comma-separated product capabilities"),
    sender_name: Optional[str] = typer.Option(None, "--sender-name", help="Sender given name"),
    sender_company: Optional[str] = typer.Option(None, "--sender-company", help="Sender organization"),
    sender_identity: Optional[str] = typer.Option(None, "--sender-identity", help="Sender full identity"),
    objective: Optional[str] = typer.Option(None, "--objective", help="Outreach objective"),
    max_iterations: int = typer.Option(3, "--max-iterations", help="Max research iterations"),
    allow_insufficient: bool = typer.Option(False, "--allow-insufficient", help="Allow proceeding to strategy even if research marked insufficient"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Disallow interactive input prompting"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Custom SQLite database path"),
):
    """Execute the end-to-end Zamp outreach pipeline for a prospect."""
    if ctx.invoked_subcommand is not None:
        return

    # 1. Parse from input file if provided
    prospect_data: Dict[str, Any] = {}
    product_data: Dict[str, Any] = {}
    outreach_data: Dict[str, Any] = {}

    if input_file:
        if not input_file.exists():
            console.print(f"[bold red]Error: Input file {input_file} does not exist.[/bold red]")
            raise typer.Exit(1)
        try:
            content = json.loads(input_file.read_text(encoding="utf-8"))
            prospect_data = content.get("prospect", {})
            product_data = content.get("product", {})
            outreach_data = content.get("outreach", {})
        except Exception as err:
            console.print(f"[bold red]Failed to parse JSON input file:[/bold red] {err}")
            raise typer.Exit(1)

    # 2. Reconcile parameters with CLI overrides or interactive prompts
    p_name = name or prospect_data.get("full_name")
    p_company = company or prospect_data.get("company_name")
    p_domain = domain or prospect_data.get("company_domain")
    p_role = designation or prospect_data.get("designation")
    p_linkedin = linkedin or prospect_data.get("linkedin_url")

    prod_name = product_name or product_data.get("product_name") or "Zamp Autonomous Engine"
    prod_desc = product_desc or product_data.get("description") or "Autonomous evidence-grounded research and outreach platform."
    prod_vp = value_prop or product_data.get("value_proposition") or "Eliminate cold outreach spam through verified commercial signals and deep research."
    caps_list = [c.strip() for c in capabilities.split(",")] if capabilities else product_data.get("capabilities", [
        "Real-time company intelligence",
        "Multi-turn web evidence corroboration",
        "Automated commercial signal identification",
    ])

    out_obj = objective or outreach_data.get("objective") or "Explore strategic partnership and pipeline expansion"
    s_name = sender_name or outreach_data.get("sender_name") or "Alex Rivers"
    s_comp = sender_company or outreach_data.get("sender_company") or "Zamp AI"
    s_ident = sender_identity or outreach_data.get("sender_identity") or f"{s_name}, Solutions Architect at {s_comp}"

    if not p_name or not p_company:
        if non_interactive or not sys.stdin.isatty():
            console.print("[bold red]Error: Missing required prospect full_name or company_name in non-interactive mode.[/bold red]")
            raise typer.Exit(1)
        p_name = typer.prompt("Enter prospect full name", default=p_name)
        p_company = typer.prompt("Enter prospect company name", default=p_company)

    prospect = ProspectContext(
        full_name=p_name,
        company_name=p_company,
        company_domain=p_domain,
        designation=p_role,
        linkedin_url=p_linkedin,
    )
    product = ProductContext(
        product_name=prod_name,
        description=prod_desc,
        capabilities=caps_list,
        value_proposition=prod_vp,
    )
    outreach = OutreachContext(
        objective=out_obj,
        sender_identity=s_ident,
        sender_name=s_name,
        sender_company=s_comp,
    )

    orchestrator = _get_orchestrator(db_path)

    console.print(Panel(
        f"[bold white]Target Prospect:[/bold white] {prospect.full_name} ({prospect.designation or 'Role N/A'}) at [bold cyan]{prospect.company_name}[/bold cyan]\n"
        f"[bold white]Offering:[/bold white] {product.product_name} - {product.value_proposition}\n"
        f"[bold white]Max Iterations:[/bold white] {max_iterations}",
        title="[bold green]Starting Zamp Outreach Pipeline[/bold green]",
        border_style="green",
    ))

    # Execute pipeline with spinner
    with console.status("[bold green]Executing multi-stage pipeline...[/bold green]", spinner="dots"):
        result: RunResult = orchestrator.run(
            prospect=prospect,
            product=product,
            outreach=outreach,
            max_research_iterations=max_iterations,
            allow_insufficient_research=allow_insufficient,
        )

    # Render Outcome
    _render_run_result(result)


def _render_run_result(result: RunResult):
    status_color = "green" if result.status == RunStatus.COMPLETED else (
        "yellow" if result.status == RunStatus.COMPLETED_WITH_WARNINGS else "red"
    )

    console.print(Panel(
        f"[bold]Run ID:[/bold] {result.run_id}\n"
        f"[bold]Status:[/bold] [{status_color}]{result.status.value.upper()}[/{status_color}]\n"
        f"[bold]Duration:[/bold] {result.started_at} -> {result.completed_at}",
        title="[bold]Run Outcome Summary[/bold]",
        border_style=status_color,
    ))

    # Stages table
    table = Table(title="Stage Execution Breakdown", show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Duration", justify="right")
    table.add_column("Details", style="dim")

    for s in result.stage_records:
        st_color = "green" if s.status == "completed" else ("red" if s.status == "failed" else "yellow")
        details_str = json.dumps(s.details) if s.details else (s.error_message or "")
        table.add_row(s.stage.value, f"[{st_color}]{s.status.value.upper()}[/{st_color}]", f"{s.duration_ms} ms", details_str[:80])

    console.print(table)

    # Strategy result
    if result.strategy_result and result.strategy_result.strategy:
        strat = result.strategy_result.strategy
        sig = strat.primary_signal

        strat_text = (
            f"[bold cyan]Messaging Approach:[/bold cyan] {strat.messaging_approach}\n"
            f"[bold cyan]Central Value Proposition:[/bold cyan] {strat.central_message}\n"
            f"[bold cyan]Suggested Subject Line:[/bold cyan] {strat.suggested_subject_line}\n"
            f"[bold cyan]Call to Action:[/bold cyan] {strat.cta}\n"
            f"[bold cyan]Tone Guidance:[/bold cyan] {strat.tone_guidance}\n\n"
            f"[bold white]Selected Commercial Hook:[/bold white] [bold yellow]{sig.signal.label}[/bold yellow] (Score: {sig.score.composite_score:.2f})\n"
            f"[dim]\"{sig.signal.description}\"[/dim]\n\n"
            f"[bold white]Product Capabilities Mapped:[/bold white]\n"
        )
        for cap in strat.alignment.capability_mapping:
            strat_text += f"  • {cap}\n"

        console.print(Panel(strat_text.strip(), title="[bold green]Formulated Outreach Strategy[/bold green]", border_style="green"))
    elif result.strategy_result and not result.strategy_result.is_viable:
        fail_text = (
            f"[bold red]Rejection Reason:[/bold red] {result.strategy_result.rejection_reason}\n\n"
            f"[bold white]Diagnostic Non-viability Reasons:[/bold white]\n"
        )
        for r in result.strategy_result.unviable_reasons:
            fail_text += f"  • {r}\n"
        console.print(Panel(fail_text.strip(), title="[bold red]Strategy Not Viable[/bold red]", border_style="red"))
    elif result.error_message:
        console.print(Panel(f"[bold red]Failure Message:[/bold red] {result.error_message}", title="[bold red]Pipeline Failed[/bold red]", border_style="red"))

    if result.warnings:
        w_text = "\n".join([f"• {w}" for w in result.warnings])
        console.print(Panel(w_text, title="[bold yellow]Warnings[/bold yellow]", border_style="yellow"))


@run_app.command("list")
def run_list(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of runs to display"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by RunStatus"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Custom SQLite database path"),
):
    """List historical application runs."""
    orchestrator = _get_orchestrator(db_path)
    runs = orchestrator.list_runs(limit=limit, status=RunStatus(status) if status else None)

    if not runs:
        console.print("[yellow]No application runs found.[/yellow]")
        return

    table = Table(title=f"Application Runs (Latest {len(runs)})", show_header=True, header_style="bold cyan")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Prospect", style="white")
    table.add_column("Company", style="white")
    table.add_column("Started At", style="dim")

    for r in runs:
        st = r["status"]
        st_color = "green" if st == "completed" else ("yellow" if "warning" in st else "red")
        p_json = json.loads(r["prospect_snapshot_json"]) if r.get("prospect_snapshot_json") else {}
        table.add_row(
            r["run_id"][:12] + "...",
            f"[{st_color}]{st.upper()}[/{st_color}]",
            p_json.get("full_name", "N/A"),
            p_json.get("company_name", "N/A"),
            r["started_at"][:19] if r.get("started_at") else "N/A",
        )

    console.print(table)


@run_app.command("inspect")
def run_inspect(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Custom SQLite database path"),
):
    """Inspect complete details and strategy blueprint for a specific application run."""
    orchestrator = _get_orchestrator(db_path)
    snapshot = orchestrator.get_run(run_id)

    if not snapshot:
        # Try prefix matching
        runs = orchestrator.list_runs(limit=100)
        matched = [r for r in runs if r["run_id"].startswith(run_id)]
        if matched:
            snapshot = orchestrator.get_run(matched[0]["run_id"])

    if not snapshot:
        console.print(f"[bold red]Run not found for ID:[/bold red] {run_id}")
        raise typer.Exit(1)

    status_color = "green" if snapshot.status == RunStatus.COMPLETED else (
        "yellow" if snapshot.status == RunStatus.COMPLETED_WITH_WARNINGS else "red"
    )

    grid = Table.grid(padding=1)
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column(style="white")

    grid.add_row("Run ID:", snapshot.run_id)
    grid.add_row("Status:", f"[{status_color}]{snapshot.status.value.upper()}[/{status_color}]")
    grid.add_row("Prospect:", f"{snapshot.prospect.full_name} ({snapshot.prospect.designation or 'N/A'})")
    grid.add_row("Company:", snapshot.prospect.company_name)
    grid.add_row("Product:", snapshot.product.product_name)
    grid.add_row("Session ID:", str(snapshot.research_session_id))
    grid.add_row("Started At:", snapshot.started_at)
    grid.add_row("Completed At:", str(snapshot.completed_at))

    console.print(Panel(grid, title=f"[bold]Run {snapshot.run_id}[/bold]", border_style=status_color))

    if snapshot.strategy_result and snapshot.strategy_result.strategy:
        strat = snapshot.strategy_result.strategy
        sig = strat.primary_signal

        content = (
            f"[bold cyan]Messaging Approach:[/bold cyan] {strat.messaging_approach}\n"
            f"[bold cyan]Core Value Proposition:[/bold cyan] {strat.central_message}\n"
            f"[bold cyan]Subject Line:[/bold cyan] {strat.suggested_subject_line}\n"
            f"[bold cyan]Call to Action:[/bold cyan] {strat.cta}\n"
            f"[bold cyan]Tone Guidance:[/bold cyan] {strat.tone_guidance}\n\n"
            f"[bold white]Selected Commercial Signal:[/bold white] [bold yellow]{sig.signal.label}[/bold yellow] (Composite Score: {sig.score.composite_score:.2f})\n"
            f"[dim]\"{sig.signal.description}\"[/dim]\n\n"
            f"[bold white]Capability Mappings:[/bold white]\n"
        )
        for m in strat.alignment.capability_mapping:
            content += f"  • {m}\n"
        console.print(Panel(content.strip(), title="[bold green]Selected Strategy[/bold green]", border_style="green"))
    elif snapshot.strategy_result and not snapshot.strategy_result.is_viable:
        console.print(Panel(f"[bold red]Rejection Reason:[/bold red] {snapshot.strategy_result.rejection_reason}", title="[bold red]Strategy Unviable[/bold red]", border_style="red"))

    if snapshot.warnings:
        console.print(Panel("\n".join([f"• {w}" for w in snapshot.warnings]), title="[bold yellow]Warnings[/bold yellow]", border_style="yellow"))


@run_app.command("stages")
def run_stages(
    run_id: str = typer.Argument(..., help="Run ID to inspect stages"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Custom SQLite database path"),
):
    """Inspect stage execution timeline and duration breakdown for a run."""
    orchestrator = _get_orchestrator(db_path)
    stages = orchestrator.get_run_stages(run_id)

    if not stages:
        # Try prefix matching
        runs = orchestrator.list_runs(limit=100)
        matched = [r for r in runs if r["run_id"].startswith(run_id)]
        if matched:
            stages = orchestrator.get_run_stages(matched[0]["run_id"])

    if not stages:
        console.print(f"[bold red]No stage records found for run ID:[/bold red] {run_id}")
        raise typer.Exit(1)

    table = Table(title=f"Stage Timeline for Run {run_id}", show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Started At", style="dim")
    table.add_column("Duration", justify="right")
    table.add_column("Details", style="dim")

    for s in stages:
        st_color = "green" if s.status == "completed" else ("red" if s.status == "failed" else "yellow")
        d_str = json.dumps(s.details) if s.details else (s.error_message or "")
        table.add_row(
            s.stage.value,
            f"[{st_color}]{s.status.value.upper()}[/{st_color}]",
            s.started_at[11:19] if len(s.started_at) > 19 else s.started_at,
            f"{s.duration_ms} ms",
            d_str[:70],
        )

    console.print(table)


# =============================================================================
# Strategy Explainability & Research Inspection
# =============================================================================

@strategy_app.command("inspect")
def strategy_inspect(
    run_id: str = typer.Argument(..., help="Run ID to explain strategy evidence provenance"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Custom SQLite database path"),
):
    """Inspect domain explainability tree connecting selected strategy to supporting facts, excerpts, and URLs."""
    orchestrator = _get_orchestrator(db_path)
    explain = orchestrator.get_strategy_explainability(run_id)

    if "error" in explain:
        # Try prefix matching
        runs = orchestrator.list_runs(limit=100)
        matched = [r for r in runs if r["run_id"].startswith(run_id)]
        if matched:
            explain = orchestrator.get_strategy_explainability(matched[0]["run_id"])

    if "error" in explain:
        console.print(f"[bold red]{explain['error']}[/bold red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]Prospect:[/bold cyan] {explain['prospect_name']} ({explain['company_name']})\n"
        f"[bold cyan]Approach:[/bold cyan] {explain['messaging_approach']}\n"
        f"[bold cyan]Core Value Proposition:[/bold cyan] {explain['central_message']}\n"
        f"[bold cyan]Selected Hook:[/bold cyan] [bold yellow]{explain['primary_signal']['signal']['label']}[/bold yellow] (Score: {explain['primary_signal']['score']['composite_score']:.2f})",
        title="[bold green]Strategy Overview[/bold green]",
        border_style="green",
    ))

    # Evidence chain table
    if explain["evidence_chain"]:
        table = Table(title="Corroborating Evidence Passages & Source Provenance", show_header=True, header_style="bold cyan")
        table.add_column("Fact ID", style="dim", width=10)
        table.add_column("Factual Claim", style="white")
        table.add_column("Verbatim Source Excerpt", style="yellow")
        table.add_column("Channel / Domain", style="cyan")
        table.add_column("Credibility", justify="right")

        for ev in explain["evidence_chain"]:
            table.add_row(
                ev["fact_id"][:8],
                ev["claim_text"][:50],
                f"\"{ev['verbatim_excerpt'][:80]}\"",
                ev["domain"] or ev["source_url"] or "N/A",
                f"{ev['credibility_score']:.2f}",
            )
        console.print(table)
    else:
        console.print("[dim]No underlying evidence rows found in database for this session.[/dim]")


@research_app.command("inspect")
def research_inspect(
    session_id: str = typer.Argument(..., help="Research session ID to inspect"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Custom SQLite database path"),
):
    """Inspect research objectives, discovered sources, canonical facts, and conflicts for a session."""
    orchestrator = _get_orchestrator(db_path)
    data = orchestrator.get_research_inspection(session_id)

    if not data:
        console.print(f"[bold red]Session or run not found for ID:[/bold red] {session_id}")
        raise typer.Exit(1)

    objectives = data["objectives"]
    facts = data["facts"]
    conflicts = data["conflicts"]

    # Summary Panel
    s_color = "green" if data.get("is_sufficient") else "yellow"
    console.print(Panel(
        f"[bold]Session ID:[/bold] {data['session_id']}\n"
        f"[bold]Status:[/bold] {data['status']}\n"
        f"[bold]Sufficient:[/bold] [{s_color}]{data.get('is_sufficient')}[/{s_color}]\n"
        f"[bold]Completed:[/bold] {data.get('is_complete')}\n"
        f"[bold]Sufficiency Notes:[/bold] {data.get('sufficiency_notes') or 'N/A'}",
        title=f"[bold]Research Session {data['session_id']}[/bold]",
        border_style=s_color,
    ))

    # Objectives Table
    if objectives:
        o_table = Table(title="Research Objectives", header_style="bold cyan")
        o_table.add_column("Type", style="cyan")
        o_table.add_column("Target", style="white")
        o_table.add_column("Description", style="white")
        o_table.add_column("Priority", justify="right")
        o_table.add_column("Critical", justify="center")

        for o in objectives:
            o_table.add_row(
                o["type"],
                o["target_entity"],
                o["description"],
                str(o["priority"]),
                "[red]YES[/red]" if o.get("is_critical") else "No",
            )
        console.print(o_table)

    # Canonical Facts Table
    if facts:
        f_table = Table(title=f"Canonical Facts Extracted ({len(facts)})", header_style="bold green")
        f_table.add_column("Fact ID", style="dim", width=10)
        f_table.add_column("Category", style="cyan")
        f_table.add_column("Claim Text", style="white")
        f_table.add_column("Confidence", style="bold")
        f_table.add_column("Event Date", style="dim")

        for f in facts:
            f_table.add_row(
                f["fact_id"][:8],
                f["fact_category"],
                f["claim_text"],
                f["composite_confidence"],
                str(f.get("primary_event_date") or "N/A"),
            )
        console.print(f_table)

    if conflicts:
        c_table = Table(title=f"Detected Contradictions / Conflicts ({len(conflicts)})", header_style="bold red")
        c_table.add_column("Conflict ID", style="dim")
        c_table.add_column("Severity", style="bold red")
        c_table.add_column("Description", style="white")
        c_table.add_column("Status", style="yellow")
        for c in conflicts:
            c_table.add_row(c["conflict_id"][:8], c.get("severity", "N/A"), c["description"], c["status"])
        console.print(c_table)



@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Custom SQLite database path"),
):
    """Start Zamp HTTP API server."""
    init_env()
    if db_path:
        os.environ["ZAMP_DB_PATH"] = db_path
    import uvicorn

    console.print(
        Panel(
            f"[bold cyan]Starting Zamp API Server[/bold cyan]\n"
            f"Address: http://{host}:{port}\n"
            f"Database: {get_db_path()}",
            border_style="cyan",
        )
    )
    uvicorn.run("src.api.app:create_app", host=host, port=port, reload=reload, factory=True)


if __name__ == "__main__":
    app()

