"""Build PDFs of the two write-ups with pandoc + xelatex.

The committed markdown follows the repo's Obsidian conventions: YAML frontmatter,
no H1, `[[wikilinks]]`. Three of those are wrong for a standalone PDF, so this
preprocesses a copy rather than changing the canonical source:

  - frontmatter `author`/`date` become pandoc metadata; `tags` is dropped
  - the first `## ` heading is removed and promoted to the document title,
    otherwise pandoc prints it twice (once as \\title, once as a section)
  - `[[name]]` becomes italic `name`, since pandoc renders wikilinks literally

Requires pandoc (winget: JohnMacFarlane.Pandoc) and a LaTeX engine (MiKTeX
supplies xelatex here). xelatex is used rather than pdflatex because the
documents contain em/en dashes, U+2212 MINUS, ->, +- and x.

Run: python scripts/25_build_pdfs.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path("docs/insen")
JOBS = [
    ("fra_progress_summary", "FRA against ground truth: progress summary", False),
    ("fra_technical_report", "FRA against ground truth: technical report", True),
]


def preprocess(src: Path) -> tuple[str, dict[str, str]]:
    text = src.read_text(encoding="utf-8")
    meta: dict[str, str] = {}

    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        for line in m.group(1).split("\n"):
            k, _, v = line.partition(":")
            if k.strip() in ("author", "date") and v.strip():
                meta[k.strip()] = v.strip()
        text = text[m.end():]

    # Drop the first H2; it duplicates the title we pass as metadata.
    text = re.sub(r"^\s*## .*?\n", "", text, count=1)

    # Obsidian wikilinks -> italics.
    text = re.sub(r"\[\[([^\]|]+)\]\]", lambda m: f"*{m.group(1)}*", text)

    return text.lstrip("\n"), meta


def main() -> int:
    if shutil.which("pandoc") is None:
        print("pandoc not on PATH", file=sys.stderr)
        return 1

    for stem, title, toc in JOBS:
        src = DOCS / f"{stem}.md"
        out = DOCS / f"{stem}.pdf"
        body, meta = preprocess(src)

        with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8",
                                         delete=False, dir=DOCS) as fh:
            fh.write(body)
            tmp = Path(fh.name)

        cmd = [
            "pandoc", str(tmp), "-o", str(out),
            "--pdf-engine=xelatex",
            "--shift-heading-level-by=-1",
            "--resource-path", str(DOCS),
            "-V", "geometry:margin=2.5cm",
            "-V", "fontsize=11pt",
            "-V", "colorlinks=true",
            # LaTeX verbatim does not wrap, so long lines in fenced code run off
            # the page and are silently truncated. fvextra's breaklines fixes it;
            # the widest line in these documents is 100 characters.
            "-V", "header-includes="
                  r"\usepackage{fvextra}"
                  r"\DefineVerbatimEnvironment{Highlighting}{Verbatim}"
                  r"{breaklines,breakanywhere,commandchars=\\\{\}}",
            "-M", f"title={title}",
        ]
        for k, v in meta.items():
            cmd += ["-M", f"{k}={v}"]
        if toc:
            cmd += ["--toc", "--toc-depth=3"]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            tmp.unlink(missing_ok=True)

        warn = [l for l in (r.stderr or "").split("\n") if "Missing character" in l]
        print(f"{stem}: exit={r.returncode}  {out.stat().st_size // 1024 if out.exists() else 0}KB  "
              f"missing-glyph warnings={len(warn)}")
        for l in warn[:5]:
            print(f"    {l.strip()}")
        if r.returncode != 0:
            print((r.stderr or "")[-1500:], file=sys.stderr)
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
