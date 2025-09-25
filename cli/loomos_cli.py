#!/usr/bin/env python3
"""
LoomOS Command Line Interface

Production-ready CLI for LoomOS with comprehensive functionality:
- Job submission and management
- Real-time log streaming
- Cluster monitoring and administration
- Worker management
- Marketplace operations
- Configuration management
"""

import asyncio
import json
import yaml
import time
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
import websockets
from urllib.parse import urlparse

# Initialize CLI app
app = typer.Typer(
    name="loomos",
    help="LoomOS - The Iron Suit for AI Models",
    rich_markup_mode="rich"
)
console = Console()

# Global configuration
class Config:
    def __init__(self):
        self.api_base = os.getenv("LOOMOS_API_URL", "http://localhost:8000")
        self.token = os.getenv("LOOMOS_TOKEN", "")
        self.timeout = int(os.getenv("LOOMOS_TIMEOUT", "300"))
        self.config_dir = Path.home() / ".loomos"
        self.config_file = self.config_dir / "config.yaml"
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    config_data = yaml.safe_load(f)
                    self.api_base = config_data.get("api_base", self.api_base)
                    self.token = config_data.get("token", self.token)
                    self.timeout = config_data.get("timeout", self.timeout)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load config: {e}[/yellow]")
    
    def save_config(self):
        """Save configuration to file"""
        self.config_dir.mkdir(exist_ok=True)
        config_data = {
            "api_base": self.api_base,
            "token": self.token,
            "timeout": self.timeout
        }
        with open(self.config_file, "w") as f:
            yaml.safe_dump(config_data, f)

config = Config()

# HTTP client setup
def get_client() -> httpx.AsyncClient:
    """Get configured HTTP client"""
    headers = {"Authorization": f"Bearer {config.token}"} if config.token else {}
    return httpx.AsyncClient(
        base_url=config.api_base,
        headers=headers,
        timeout=config.timeout
    )

# Job management commands
@app.group(name="job")
def job_commands():
    """Job management commands"""
    pass

@job_commands.command("submit")
def submit_job(
    manifest: str = typer.Argument(..., help="Path to job manifest file"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for job completion"),
    follow_logs: bool = typer.Option(False, "--logs", "-l", help="Follow job logs"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
):
    """Submit a job from manifest file"""
    asyncio.run(submit_job_async(manifest, wait, follow_logs, output_format))

async def submit_job_async(manifest: str, wait: bool, follow_logs: bool, output_format: str):
    """Async job submission"""
    try:
        # Load manifest
        manifest_path = Path(manifest)
        if not manifest_path.exists():
            console.print(f"[red]Error: Manifest file not found: {manifest}[/red]")
            raise typer.Exit(1)
        
        with open(manifest_path) as f:
            if manifest.endswith('.yaml') or manifest.endswith('.yml'):
                job_spec = yaml.safe_load(f)
            else:
                job_spec = json.load(f)
        
        # Submit job
        async with get_client() as client:
            with console.status("[bold blue]Submitting job..."):
                response = await client.post("/v1/jobs", json=job_spec)
                response.raise_for_status()
                result = response.json()
            
            job_id = result["job_id"]
            
            if output_format == "json":
                console.print(json.dumps(result, indent=2))
            elif output_format == "yaml":
                console.print(yaml.dump(result, default_flow_style=False))
            else:
                console.print(f"[green]✓[/green] Job submitted successfully")
                console.print(f"[bold]Job ID:[/bold] {job_id}")
                console.print(f"[bold]Status:[/bold] {result['status']}")
            
            # Wait and/or follow logs if requested
            if wait or follow_logs:
                if follow_logs:
                    await stream_logs_async(job_id)
                if wait:
                    await wait_for_completion(job_id)
    
    except httpx.RequestError as e:
        console.print(f"[red]Error: Failed to connect to LoomOS API: {e}[/red]")
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: API request failed ({e.response.status_code}): {e.response.text}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@job_commands.command("list")
def list_jobs(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum number of jobs to show"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
):
    """List jobs"""
    asyncio.run(list_jobs_async(status, limit, output_format))

async def list_jobs_async(status: Optional[str], limit: int, output_format: str):
    """Async job listing"""
    try:
        params = {"limit": limit}
        if status:
            params["status"] = status
        
        async with get_client() as client:
            with console.status("[bold blue]Fetching jobs..."):
                response = await client.get("/v1/jobs", params=params)
                response.raise_for_status()
                jobs = response.json()
        
        if output_format == "json":
            console.print(json.dumps(jobs, indent=2))
        elif output_format == "yaml":
            console.print(yaml.dump(jobs, default_flow_style=False))
        else:
            # Table format
            table = Table(title="Jobs")
            table.add_column("Job ID", style="cyan")
            table.add_column("Name", style="bold")
            table.add_column("Status", style="magenta")
            table.add_column("Progress", style="green")
            table.add_column("Created", style="dim")
            
            for job in jobs:
                progress = f"{job.get('progress', 0):.1%}" if job.get('progress') is not None else "N/A"
                created = job.get('created_at', '').replace('T', ' ').split('.')[0] if job.get('created_at') else 'N/A'
                table.add_row(
                    job.get('job_id', 'N/A'),
                    job.get('name', 'N/A'),
                    job.get('status', 'N/A'),
                    progress,
                    created
                )
            
            console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@job_commands.command("status")
def job_status(
    job_id: str = typer.Argument(..., help="Job ID"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
):
    """Get job status"""
    asyncio.run(job_status_async(job_id, output_format))

async def job_status_async(job_id: str, output_format: str):
    """Async job status"""
    try:
        async with get_client() as client:
            with console.status(f"[bold blue]Fetching status for {job_id}..."):
                response = await client.get(f"/v1/jobs/{job_id}")
                response.raise_for_status()
                job = response.json()
        
        if output_format == "json":
            console.print(json.dumps(job, indent=2))
        elif output_format == "yaml":
            console.print(yaml.dump(job, default_flow_style=False))
        else:
            # Rich formatted output
            panel_content = f"""[bold]Job ID:[/bold] {job.get('job_id', 'N/A')}
[bold]Name:[/bold] {job.get('name', 'N/A')}
[bold]Status:[/bold] {job.get('status', 'N/A')}
[bold]Progress:[/bold] {job.get('progress', 0):.1%}
[bold]Message:[/bold] {job.get('message', 'N/A')}
[bold]Created:[/bold] {job.get('created_at', 'N/A')}
[bold]Started:[/bold] {job.get('started_at', 'N/A')}
[bold]Completed:[/bold] {job.get('completed_at', 'N/A')}"""
            
            if job.get('error'):
                panel_content += f"\n[bold red]Error:[/bold red] {job['error']}"
            
            console.print(Panel(panel_content, title="Job Status", border_style="blue"))
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@job_commands.command("cancel")
def cancel_job(job_id: str = typer.Argument(..., help="Job ID")):
    """Cancel a job"""
    asyncio.run(cancel_job_async(job_id))

async def cancel_job_async(job_id: str):
    """Async job cancellation"""
    try:
        async with get_client() as client:
            with console.status(f"[bold red]Cancelling job {job_id}..."):
                response = await client.delete(f"/v1/jobs/{job_id}")
                response.raise_for_status()
                result = response.json()
            
            console.print(f"[green]✓[/green] {result['message']}")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@job_commands.command("logs")
def job_logs(
    job_id: str = typer.Argument(..., help="Job ID"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    tail: Optional[int] = typer.Option(None, "--tail", help="Number of lines to show from end")
):
    """Get job logs"""
    asyncio.run(stream_logs_async(job_id, follow, tail))

async def stream_logs_async(job_id: str, follow: bool = True, tail: Optional[int] = None):
    """Stream job logs"""
    try:
        async with get_client() as client:
            async with client.stream("GET", f"/v1/jobs/{job_id}/logs") as response:
                response.raise_for_status()
                
                console.print(f"[bold blue]Streaming logs for job {job_id}...[/bold blue]")
                console.print("[dim]Press Ctrl+C to stop[/dim]")
                console.print("─" * console.width)
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            log_entry = json.loads(line[6:])  # Remove "data: " prefix
                            timestamp = log_entry.get('timestamp', '')
                            level = log_entry.get('level', 'INFO')
                            message = log_entry.get('message', '')
                            
                            # Color code log levels
                            level_colors = {
                                'DEBUG': 'dim',
                                'INFO': 'blue',
                                'WARNING': 'yellow',
                                'ERROR': 'red',
                                'CRITICAL': 'bold red'
                            }
                            level_color = level_colors.get(level, 'white')
                            
                            console.print(f"[dim]{timestamp}[/dim] [{level_color}]{level}[/{level_color}] {message}")
                        except json.JSONDecodeError:
                            console.print(line)
                    
                    if not follow:
                        break
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Log streaming stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

async def wait_for_completion(job_id: str):
    """Wait for job completion"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Waiting for job {job_id}", total=100)
        
        async with get_client() as client:
            while True:
                try:
                    response = await client.get(f"/v1/jobs/{job_id}")
                    response.raise_for_status()
                    job = response.json()
                    
                    status = job.get('status', 'unknown')
                    job_progress = job.get('progress', 0) * 100
                    
                    progress.update(task, completed=job_progress, description=f"Job {job_id} - {status}")
                    
                    if status in ['completed', 'failed', 'cancelled']:
                        break
                    
                    await asyncio.sleep(5)
                
                except Exception as e:
                    console.print(f"[red]Error checking job status: {e}[/red]")
                    break
        
        # Final status
        if status == 'completed':
            console.print(f"[green]✓[/green] Job {job_id} completed successfully")
        elif status == 'failed':
            console.print(f"[red]✗[/red] Job {job_id} failed")
        else:
            console.print(f"[yellow]![/yellow] Job {job_id} was cancelled")

# Cluster management commands
@app.group(name="cluster")
def cluster_commands():
    """Cluster management commands"""
    pass

@cluster_commands.command("status")
def cluster_status(
    output_format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
):
    """Get cluster status"""
    asyncio.run(cluster_status_async(output_format))

async def cluster_status_async(output_format: str):
    """Async cluster status"""
    try:
        async with get_client() as client:
            with console.status("[bold blue]Fetching cluster status..."):
                response = await client.get("/v1/cluster/stats")
                response.raise_for_status()
                stats = response.json()
        
        if output_format == "json":
            console.print(json.dumps(stats, indent=2))
        elif output_format == "yaml":
            console.print(yaml.dump(stats, default_flow_style=False))
        else:
            # Rich dashboard
            table = Table(title="Cluster Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold")
            
            metrics = [
                ("Total Workers", stats.get('total_workers', 0)),
                ("Active Workers", stats.get('active_workers', 0)),
                ("Total Jobs", stats.get('total_jobs', 0)),
                ("Active Jobs", stats.get('active_jobs', 0)),
                ("Completed Jobs", stats.get('completed_jobs', 0)),
                ("Failed Jobs", stats.get('failed_jobs', 0)),
                ("CPU Utilization", f"{stats.get('cpu_utilization', 0):.1%}"),
                ("GPU Utilization", f"{stats.get('gpu_utilization', 0):.1%}"),
                ("Memory Utilization", f"{stats.get('memory_utilization', 0):.1%}"),
                ("Avg Job Duration", f"{stats.get('avg_job_duration_seconds', 0):.1f}s"),
                ("Jobs/Hour", f"{stats.get('jobs_per_hour', 0):.1f}")
            ]
            
            for metric, value in metrics:
                table.add_row(metric, str(value))
            
            console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@cluster_commands.command("workers")
def list_workers(
    output_format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
):
    """List cluster workers"""
    asyncio.run(list_workers_async(output_format))

async def list_workers_async(output_format: str):
    """Async worker listing"""
    try:
        async with get_client() as client:
            with console.status("[bold blue]Fetching workers..."):
                response = await client.get("/v1/workers")
                response.raise_for_status()
                workers = response.json()
        
        if output_format == "json":
            console.print(json.dumps(workers, indent=2))
        elif output_format == "yaml":
            console.print(yaml.dump(workers, default_flow_style=False))
        else:
            table = Table(title="Cluster Workers")
            table.add_column("Worker ID", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Load", style="green")
            table.add_column("GPUs", style="yellow")
            table.add_column("Memory", style="blue")
            table.add_column("Last Seen", style="dim")
            
            for worker in workers:
                capabilities = worker.get('capabilities', {})
                gpu_count = capabilities.get('gpu_count', 0)
                memory_gb = capabilities.get('memory_gb', 0)
                load = f"{worker.get('current_load', 0):.1%}"
                last_seen = worker.get('last_heartbeat', '').replace('T', ' ').split('.')[0] if worker.get('last_heartbeat') else 'N/A'
                
                table.add_row(
                    worker.get('worker_id', 'N/A'),
                    worker.get('status', 'N/A'),
                    load,
                    str(gpu_count),
                    f"{memory_gb}GB",
                    last_seen
                )
            
            console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

# Configuration commands
@app.group(name="config")
def config_commands():
    """Configuration management"""
    pass

@config_commands.command("set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value")
):
    """Set configuration value"""
    if key == "api_base":
        config.api_base = value
    elif key == "token":
        config.token = value
    elif key == "timeout":
        config.timeout = int(value)
    else:
        console.print(f"[red]Unknown configuration key: {key}[/red]")
        raise typer.Exit(1)
    
    config.save_config()
    console.print(f"[green]✓[/green] Set {key} = {value}")

@config_commands.command("get")
def get_config(key: Optional[str] = typer.Argument(None, help="Configuration key")):
    """Get configuration value(s)"""
    if key:
        if key == "api_base":
            console.print(config.api_base)
        elif key == "token":
            console.print(config.token if config.token else "[not set]")
        elif key == "timeout":
            console.print(str(config.timeout))
        else:
            console.print(f"[red]Unknown configuration key: {key}[/red]")
            raise typer.Exit(1)
    else:
        table = Table(title="Configuration")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="bold")
        
        table.add_row("api_base", config.api_base)
        table.add_row("token", config.token if config.token else "[not set]")
        table.add_row("timeout", str(config.timeout))
        
        console.print(table)

@config_commands.command("login")
def login(
    api_url: str = typer.Option(None, "--api-url", help="LoomOS API URL"),
    token: str = typer.Option(None, "--token", help="Authentication token")
):
    """Login to LoomOS"""
    if api_url:
        config.api_base = api_url
    
    if not token:
        token = typer.prompt("Enter your LoomOS token", hide_input=True)
    
    config.token = token
    config.save_config()
    
    # Test connection
    asyncio.run(test_connection())

async def test_connection():
    """Test API connection"""
    try:
        async with get_client() as client:
            with console.status("[bold blue]Testing connection..."):
                response = await client.get("/v1/health")
                response.raise_for_status()
        
        console.print("[green]✓[/green] Successfully connected to LoomOS")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to connect: {e}")
        raise typer.Exit(1)

# Health and status commands
@app.command("health")
def health_check():
    """Check LoomOS system health"""
    asyncio.run(health_check_async())

async def health_check_async():
    """Async health check"""
    try:
        async with get_client() as client:
            with console.status("[bold blue]Checking system health..."):
                response = await client.get("/v1/health")
                response.raise_for_status()
                health = response.json()
        
        status = health.get('status', 'unknown')
        status_color = {
            'healthy': 'green',
            'degraded': 'yellow',
            'unhealthy': 'red'
        }.get(status, 'white')
        
        console.print(f"[bold]System Status:[/bold] [{status_color}]{status.upper()}[/{status_color}]")
        
        # Component health
        components = health.get('components', {})
        if components:
            table = Table(title="Component Health")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="bold")
            
            for component, component_health in components.items():
                component_status = component_health.get('status', 'unknown')
                table.add_row(component, component_status)
            
            console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command("version")
def version():
    """Show version information"""
    console.print("[bold blue]LoomOS CLI v1.0.0[/bold blue]")
    console.print("The Iron Suit for AI Models")

if __name__ == "__main__":
    app()