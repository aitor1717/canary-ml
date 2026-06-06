from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from canary_ml.report import DriftReport

_console = Console()


def check_alert(report: DriftReport, threshold: float) -> bool:
    """Return True when PSI exceeds *threshold*."""
    return report.psi_score > threshold


def format_alert(report: DriftReport) -> None:
    """Print a rich-formatted alert panel to the terminal."""
    drifted = sum(1 for v in report.ks_results.values() if v.get("drifted"))

    if report.alert_triggered:
        style, title = "bold red", "[bold red]DRIFT ALERT[/bold red]"
    elif report.drift_detected:
        style, title = "bold yellow", "[bold yellow]DRIFT WARNING[/bold yellow]"
    else:
        style, title = "bold green", "[bold green]STABLE[/bold green]"

    body = Text()
    body.append(f"  timestamp      ", style="dim")
    body.append(f"{report.timestamp}\n")
    body.append(f"  samples        ", style="dim")
    body.append(f"{report.n_samples}\n")
    body.append(f"  PSI score      ", style="dim")
    body.append(f"{report.psi_score:.3f}\n", style=style if report.psi_score > 0.1 else "")
    body.append(f"  features drifted ", style="dim")
    body.append(f"{drifted}\n", style=style if drifted > 0 else "")
    body.append(f"  anomaly rate   ", style="dim")
    body.append(f"{report.anomaly_rate * 100:.1f}%\n", style="yellow" if report.anomaly_rate > 0.02 else "")

    _console.print(Panel(body, title=title, border_style=style.split()[1]))
