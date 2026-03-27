"""CLI entry point: uv run dashboard → streamlit run fra/dashboard/app.py."""

def main():
    import subprocess, sys
    from pathlib import Path
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "app.py")], check=True)
