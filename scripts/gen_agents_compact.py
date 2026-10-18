#!/usr/bin/env python3
"""Generate the compact root AGENTS.md from the full text.

`.agents/reference/AGENTS.full.md` is the canonical text (Claude Code reads it through CLAUDE.md). Agent harnesses
that cap how much instruction text they load (Muse Code: 64 KiB across all instruction files) read the root
`AGENTS.md`, which is generated here: every `## ` section of the full file is copied verbatim except the ones that
have a stand-in in `.agents/reference/agents_compact_sections.md`.

    python scripts/gen_agents_compact.py                 write AGENTS.md
    python scripts/gen_agents_compact.py --check         exit 1 unless AGENTS.md is exactly what would be written
    python scripts/gen_agents_compact.py --update-hashes record the current sha of each stand-in's section, then write

A stand-in carries `sha=<12 hex>` of the full section it summarizes. When that section changes, the run fails until
someone re-reads the stand-in and runs `--update-hashes`: a stand-in cannot silently describe text that has moved on.
A stand-in whose heading is not in the full file fails too (a renamed or removed section).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / ".agents/reference/AGENTS.full.md"
STANDINS = ROOT / ".agents/reference/agents_compact_sections.md"
OUT = ROOT / "AGENTS.md"
MARKER = re.compile(r"(?m)^<!-- (header|section: (.+?) sha=([0-9a-f]{12}|PENDING)) -->\n")
FULL_REF = ".agents/reference/AGENTS.full.md"


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(preamble, [(heading, section text)]) split on `## ` lines outside fenced code blocks."""
    pre: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    fenced = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            sections.append((line[3:].strip(), [line]))
        elif sections:
            sections[-1][1].append(line)
        else:
            pre.append(line)
    return "".join(pre), [(h, "".join(ls)) for h, ls in sections]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def parse_standins(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """(header text, {heading: (recorded sha, body)})."""
    marks = list(MARKER.finditer(text))
    header = ""
    out: dict[str, tuple[str, str]] = {}
    for i, m in enumerate(marks):
        body = text[m.end() : marks[i + 1].start() if i + 1 < len(marks) else len(text)].strip("\n") + "\n"
        if m.group(1) == "header":
            header = body
        else:
            out[m.group(2)] = (m.group(3), body)
    return header, out


def build(full: str, header: str, standins: dict[str, tuple[str, str]]) -> tuple[str, list[str]]:
    """(the compact text, problems).

    A problem is a stale or orphaned stand-in.
    """
    _, sections = split_sections(full)
    present = {h for h, _ in sections}
    problems = [f"stand-in for '{h}' has no such section in {FULL_REF}" for h in standins if h not in present]
    parts = [header.rstrip("\n") + "\n\n"]
    for heading, body in sections:
        if heading in standins:
            recorded, stand = standins[heading]
            now = sha(body)
            if recorded != now:
                problems.append(
                    f"section '{heading}' changed since its stand-in was reviewed "
                    f"(recorded {recorded}, now {now}): re-read the stand-in, then run --update-hashes"
                )
            pointer = f'> Full text: `{FULL_REF}`, section "{heading}".'
            parts.append(f"## {heading} (compact)\n\n{pointer}\n\n{stand.rstrip(chr(10))}\n\n")
        else:
            parts.append(body.rstrip("\n") + "\n\n")
    return "".join(parts).rstrip("\n") + "\n", problems


def update_hashes(standin_text: str, full: str) -> str:
    _, sections = split_sections(full)
    now = {h: sha(b) for h, b in sections}
    return MARKER.sub(
        lambda m: m.group(0)
        if m.group(1) == "header" or m.group(2) not in now
        else f"<!-- section: {m.group(2)} sha={now[m.group(2)]} -->\n",
        standin_text,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update-hashes", action="store_true")
    a = ap.parse_args()
    full = FULL.read_text()
    st_text = STANDINS.read_text()
    if a.update_hashes:
        st_text = update_hashes(st_text, full)
        STANDINS.write_text(st_text)
    header, standins = parse_standins(st_text)
    if not header:
        print(f"{STANDINS}: no `<!-- header -->` block", file=sys.stderr)
        return 1
    text, problems = build(full, header, standins)
    for p in problems:
        print(f"gen_agents_compact: {p}", file=sys.stderr)
    if problems:
        return 1
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(
                "gen_agents_compact: AGENTS.md is not what the generator writes from "
                f"{FULL_REF}; run `python scripts/gen_agents_compact.py` and commit it",
                file=sys.stderr,
            )
            return 1
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(text.encode())} bytes (full text {len(full.encode())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
