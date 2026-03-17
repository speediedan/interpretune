#!/usr/bin/env python3
"""Summarize resource-debug log lines emitted by the local coverage harness.

This parser is intentionally log-format aware rather than coverage-phase aware so it can be extended later to suggest
fixture-scope changes or other optimization heuristics.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any


LOG_LINE_RE = re.compile(r"^\[(?P<prefix>[^\]]+resource_debug)\]\s(?P<label>.*?):\s(?P<body>.*)$")
GPU_KEY_RE = re.compile(r"^cuda_gpu(?P<gpu_id>\d+)_(?P<metric>.+)$")


def _coerce_value(raw_value: str) -> str | int | float | bool:
    if raw_value in {"true", "false"}:
        return raw_value == "true"
    try:
        if raw_value.isdigit() or (raw_value.startswith("-") and raw_value[1:].isdigit()):
            return int(raw_value)
        return float(raw_value)
    except ValueError:
        return raw_value


def parse_resource_log(log_path: str | Path) -> list[dict[str, Any]]:
    """Parse structured resource-debug lines from a harness log."""

    entries: list[dict[str, Any]] = []
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            match = LOG_LINE_RE.match(raw_line.strip())
            if match is None:
                continue

            fields: dict[str, Any] = {}
            try:
                tokens = shlex.split(match.group("body"))
            except ValueError:
                continue

            for token in tokens:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                fields[key] = _coerce_value(value)

            if not fields:
                continue

            entries.append(
                {
                    "line_number": line_number,
                    "prefix": match.group("prefix"),
                    "label": match.group("label"),
                    "fields": fields,
                }
            )

    return entries


def summarize_gpu_usage(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-GPU peak allocated/reserved usage across all resource lines."""

    gpu_summary: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "gpu_id": None,
            "total_gb": 0.0,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
            "max_current_allocated_gb": 0.0,
            "max_current_reserved_gb": 0.0,
        }
    )

    for entry in entries:
        for key, value in entry["fields"].items():
            match = GPU_KEY_RE.match(key)
            if match is None or not isinstance(value, (int, float)):
                continue

            gpu_id = int(match.group("gpu_id"))
            metric = match.group("metric")
            gpu_entry = gpu_summary[gpu_id]
            gpu_entry["gpu_id"] = gpu_id
            if metric == "total_gb":
                gpu_entry["total_gb"] = max(float(value), float(gpu_entry["total_gb"]))
            elif metric == "peak_allocated_gb":
                gpu_entry["peak_allocated_gb"] = max(float(value), float(gpu_entry["peak_allocated_gb"]))
            elif metric == "peak_reserved_gb":
                gpu_entry["peak_reserved_gb"] = max(float(value), float(gpu_entry["peak_reserved_gb"]))
            elif metric == "current_allocated_gb":
                gpu_entry["max_current_allocated_gb"] = max(
                    float(value),
                    float(gpu_entry["max_current_allocated_gb"]),
                )
            elif metric == "current_reserved_gb":
                gpu_entry["max_current_reserved_gb"] = max(
                    float(value),
                    float(gpu_entry["max_current_reserved_gb"]),
                )

    summary_rows: list[dict[str, Any]] = []
    for gpu_id in sorted(gpu_summary):
        row = gpu_summary[gpu_id]
        total_gb = float(row["total_gb"])
        row["peak_allocated_pct"] = (100.0 * float(row["peak_allocated_gb"]) / total_gb) if total_gb else 0.0
        row["peak_reserved_pct"] = (100.0 * float(row["peak_reserved_gb"]) / total_gb) if total_gb else 0.0
        row["max_current_reserved_pct"] = 100.0 * float(row["max_current_reserved_gb"]) / total_gb if total_gb else 0.0
        summary_rows.append(row)

    return summary_rows


def summarize_fixture_usage(entries: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    """Aggregate fixture-level VRAM observations from structured fixture log metadata."""

    fixture_summary: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for entry in entries:
        if entry["prefix"] != "fixture_resource_debug":
            continue

        fields = entry["fields"]
        if fields.get("context") != "fixture":
            continue

        fixture_id = (
            str(fields.get("kind", "unknown")),
            str(fields.get("key", "unknown")),
            str(fields.get("scope", "unknown")),
            str(fields.get("phase", "-")),
        )
        fixture_entry = fixture_summary.setdefault(
            fixture_id,
            {
                "kind": fixture_id[0],
                "key": fixture_id[1],
                "scope": fixture_id[2],
                "phase": fixture_id[3],
                "setup_reserved_delta_gb": 0.0,
                "setup_peak_reserved_delta_gb": 0.0,
                "peak_reserved_gb": 0.0,
                "peak_allocated_gb": 0.0,
                "observations": 0,
            },
        )
        fixture_entry["observations"] += 1

        for key, value in fields.items():
            if not isinstance(value, (int, float)):
                continue

            if key.startswith("delta_cuda_gpu") and key.endswith("_current_reserved_gb"):
                fixture_entry["setup_reserved_delta_gb"] = max(
                    fixture_entry["setup_reserved_delta_gb"],
                    float(value),
                )
            elif key.startswith("delta_cuda_gpu") and key.endswith("_peak_reserved_gb"):
                fixture_entry["setup_peak_reserved_delta_gb"] = max(
                    fixture_entry["setup_peak_reserved_delta_gb"],
                    float(value),
                )
            elif key.startswith("cuda_gpu") and key.endswith("_peak_reserved_gb"):
                fixture_entry["peak_reserved_gb"] = max(fixture_entry["peak_reserved_gb"], float(value))
            elif key.startswith("cuda_gpu") and key.endswith("_peak_allocated_gb"):
                fixture_entry["peak_allocated_gb"] = max(fixture_entry["peak_allocated_gb"], float(value))

    rows = sorted(
        fixture_summary.values(),
        key=lambda row: (row["setup_reserved_delta_gb"], row["peak_reserved_gb"], row["observations"]),
        reverse=True,
    )
    return rows[:limit]


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _render_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    rendered = [_render_row(headers), separator]
    rendered.extend(_render_row(row) for row in rows)
    return "\n".join(rendered)


def build_summary_text(entries: list[dict[str, Any]]) -> str:
    """Render a human-readable summary for the coverage log."""

    gpu_rows = summarize_gpu_usage(entries)
    fixture_rows = summarize_fixture_usage(entries)
    sections = ["Resource Summary"]

    if gpu_rows:
        sections.append("GPU Peak Usage")
        sections.append(
            _format_table(
                [
                    "GPU",
                    "Total GB",
                    "Peak Alloc GB",
                    "Peak Alloc %",
                    "Peak Reserved GB",
                    "Peak Reserved %",
                    "Max Current Reserved %",
                ],
                [
                    [
                        str(row["gpu_id"]),
                        f"{row['total_gb']:.2f}",
                        f"{row['peak_allocated_gb']:.2f}",
                        f"{row['peak_allocated_pct']:.1f}%",
                        f"{row['peak_reserved_gb']:.2f}",
                        f"{row['peak_reserved_pct']:.1f}%",
                        f"{row['max_current_reserved_pct']:.1f}%",
                    ]
                    for row in gpu_rows
                ],
            )
        )
    else:
        sections.append("GPU Peak Usage\nNo CUDA resource lines were found.")

    if fixture_rows:
        sections.append("Fixture VRAM Estimates")
        sections.append(
            _format_table(
                [
                    "Kind",
                    "Key",
                    "Scope",
                    "Phase",
                    "Setup Reserved Delta GB",
                    "Setup Peak Delta GB",
                    "Peak Reserved GB",
                    "Peak Alloc GB",
                    "Obs",
                ],
                [
                    [
                        row["kind"],
                        row["key"],
                        row["scope"],
                        row["phase"],
                        f"{row['setup_reserved_delta_gb']:.2f}",
                        f"{row['setup_peak_reserved_delta_gb']:.2f}",
                        f"{row['peak_reserved_gb']:.2f}",
                        f"{row['peak_allocated_gb']:.2f}",
                        str(row["observations"]),
                    ]
                    for row in fixture_rows
                ],
            )
        )
    else:
        sections.append("Fixture VRAM Estimates\nNo structured fixture resource lines were found.")

    return "\n\n".join(sections)


def build_summary_payload(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a JSON-serializable summary payload."""

    return {
        "entry_count": len(entries),
        "gpu_summary": summarize_gpu_usage(entries),
        "fixture_summary": summarize_fixture_usage(entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", required=True, help="Coverage/session log to parse")
    parser.add_argument("--json-output", help="Optional path to write a JSON summary")
    args = parser.parse_args()

    entries = parse_resource_log(args.log_file)
    payload = build_summary_payload(entries)
    print(build_summary_text(entries))
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
