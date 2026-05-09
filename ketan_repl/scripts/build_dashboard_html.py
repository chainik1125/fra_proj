"""Inline the dashboard JSON into the HTML template so it works from file://."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--data",     type=Path, required=True)
    p.add_argument("--output",   type=Path, required=True)
    args = p.parse_args()

    template = args.template.read_text()
    data = args.data.read_text()
    # Replace the loadData() fetch path with inline data.
    needle = """async function loadData() {
  const resp = await fetch("dashboard_data_50k.json");
  DATA = await resp.json();"""
    if needle not in template:
        raise SystemExit("loadData() block not found — template structure changed")
    inlined = needle.replace(
        '''async function loadData() {
  const resp = await fetch("dashboard_data_50k.json");
  DATA = await resp.json();''',
        f'''async function loadData() {{
  DATA = {data.strip()};''',
    )
    out = template.replace(needle, inlined)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out)
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
