"""CLI entry point for running the FRA Dashboard via Streamlit."""


def main():
    import subprocess
    import sys
    from pathlib import Path

    app = str(Path(__file__).parent / "__init__.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app], check=True)
