#!/usr/bin/env python
"""
Script to run FRA visualizations.

Usage:
    python run_visualization.py          # Generate HTML dashboard
    python run_visualization.py --streamlit  # Run Streamlit app (if available)
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

# Project root: parent of the directory containing this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRA_DIR = PROJECT_ROOT / "fra"


def run_html_dashboard():
    """Generate and open HTML dashboard."""
    print("Generating FRA HTML Dashboard...")

    subprocess.run(
        [sys.executable, "-m", "fra.single_sample_viz"],
        cwd=PROJECT_ROOT,
    )

    results_dir = FRA_DIR / "results"
    dashboards = list(results_dir.glob("fra_dashboard_*.html"))

    if dashboards:
        latest = max(dashboards, key=lambda p: p.stat().st_mtime)
        print(f"\nDashboard generated: {latest}")
        print(f"\nTo view the dashboard:")
        print(f"  1. Open in browser: file://{latest.absolute()}")
        print(f"  2. Or copy to your local machine and open")

        try:
            webbrowser.open(f"file://{latest.absolute()}")
        except Exception:
            pass
    else:
        print("No dashboard found. Please check for errors.")


def run_streamlit():
    """Run Streamlit app if available."""
    streamlit_path = FRA_DIR / "streamlit_app.py"

    if streamlit_path.exists():
        print("Starting Streamlit app...")
        print("\nStreamlit will run on: http://localhost:8501")
        print("Press Ctrl+C to stop\n")

        subprocess.run(["streamlit", "run", str(streamlit_path)], cwd=FRA_DIR)
    else:
        print(f"Streamlit app not found at: {streamlit_path}")
        print("Please create streamlit_app.py first")


def main():
    parser = argparse.ArgumentParser(description="Run FRA visualizations")
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Run Streamlit app instead of generating HTML dashboard",
    )

    args = parser.parse_args()

    if args.streamlit:
        run_streamlit()
    else:
        run_html_dashboard()


if __name__ == "__main__":
    main()
