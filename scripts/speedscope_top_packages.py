import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from tabulate import tabulate  # type: ignore
except Exception:
    # lightweight fallback if tabulate isn't installed
    def tabulate(rows, headers, **_kwargs):  # type: ignore
        header_line = " | ".join(headers)
        sep = "-" * max(len(header_line), 1)
        lines = [header_line, sep]
        for r in rows:
            lines.append(" | ".join(str(x) for x in r))
        return "\n".join(lines)


def parse_speedscope(path: Path, pkg_targets: list[str]):
    """Parse a speedscope JSON and print top imports by sample count.

    Args:
        path: Path to a speedscope-format JSON file.
        pkg_targets: Iterable of package name substrings to look for in frame
            names/files.
    """
    try:
        # Prefer UTF-8 text mode first.
        with path.open(mode="r", encoding="utf-8") as f:
            j = json.load(f)
    except UnicodeDecodeError:
        # If text mode with utf-8 fails, try reading bytes and decoding with common encodings.
        data = path.read_bytes()
        j = None
        for enc in ("utf-8", "latin-1"):
            try:
                s = data.decode(enc)
                j = json.loads(s)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        if j is None:
            print(f"Failed to decode or parse JSON from {path!s} with utf-8 or latin-1")
            raise SystemExit(1)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from {path!s}: {e}")
        raise SystemExit(1)

    profiles = j.get("profiles", [])
    shared = j.get("shared", {})
    frames = shared.get("frames") or (profiles[0].get("frames") if profiles else None)
    if not frames:
        print("no frames in shared or profile; listing shared keys", list(shared.keys()))
        raise SystemExit(1)

    samples = profiles[0].get("samples", []) if profiles else []
    counts = {k: 0 for k in pkg_targets}
    co = {k: defaultdict(int) for k in pkg_targets}

    for s in samples:
        idxs = s if isinstance(s, list) else [s]
        names = []
        files = []
        for i in idxs:
            if i is None:
                continue
            try:
                ii = int(i)
            except Exception:
                continue
            if not (0 <= ii < len(frames)):
                continue
            fi = frames[ii]
            names.append(fi.get("name", "") or "")
            files.append(fi.get("file", "") or "")
        joined = " ".join(filter(None, names + files))
        for k in pkg_targets:
            if k in joined:
                counts[k] += 1
                for f in files:
                    if f:
                        co[k][f] += 1

    total = len(samples)

    # Summary table
    summary_rows = [
        ("frames_count", len(frames)),
        ("total_samples", total),
    ]
    print(tabulate(summary_rows, ("metric", "value"), tablefmt="github"))
    print()

    # Per-package table with top files
    rows = []
    for k in pkg_targets:
        c = counts[k]
        pct = 100.0 * c / total if total else 0.0
        top = sorted(co[k].items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{Path(f).name} ({cnt})" for f, cnt in top) if top else ""
        rows.append((k, c, f"{pct:.1f}%", top_str))

    headers = ("package", "samples", "pct", "top_files")
    print(tabulate(rows, headers, tablefmt="github"))


def sample_stacks(
    path: Path,
    pkg_targets: list[str],
    name_filter: str | None = None,
    max_examples: int = 5,
    filter_uniq: str | None = None,
):
    """Collect example stacks for each package target.

    Args:
        path: speedscope JSON path
        pkg_targets: list of package substrings
        name_filter: optional substring to filter frame names/files (only include samples where this appears)
        max_examples: max examples per package
    """
    # load JSON (reuse robust logic)
    j = None
    try:
        with path.open(mode="r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        # fallback to simple read
        data = path.read_bytes()
        for enc in ("utf-8", "latin-1"):
            try:
                j = json.loads(data.decode(enc))
                break
            except Exception:
                j = None
        if j is None:
            print(f"Failed to load speedscope JSON for sampling stacks: {path}")
            return

    profiles = j.get("profiles", [])
    shared = j.get("shared", {})
    frames = shared.get("frames") or (profiles[0].get("frames") if profiles else None)
    if not frames:
        return

    samples = profiles[0].get("samples", []) if profiles else []
    overall_total = len(samples)

    # For each package, collect matching samples. We deduplicate by the
    # reconstructed stack lines (after applying name_filter) so we only
    # return up to `max_examples` unique stacks per package.
    pkg_examples: dict[str, list[list[str]]] = {k: [] for k in pkg_targets}
    pkg_seen: dict[str, set[tuple[str, ...]]] = {k: set() for k in pkg_targets}
    # counts per unique dedupe key per package
    pkg_key_counts: dict[str, defaultdict[tuple, int]] = {k: defaultdict(int) for k in pkg_targets}
    # total matching samples per package (used for percent calculation)
    pkg_total_counts: dict[str, int] = {k: 0 for k in pkg_targets}

    for s in samples:
        idxs = s if isinstance(s, list) else [s]
        # build lists of frame metadata for searching and possible filtering
        names: list[str] = []
        files: list[str] = []
        for i in idxs:
            if i is None:
                continue
            try:
                ii = int(i)
            except Exception:
                continue
            if not (0 <= ii < len(frames)):
                continue
            fi = frames[ii]
            names.append(fi.get("name", "") or "")
            files.append(fi.get("file", "") or "")
        joined = " ".join(filter(None, names + files))

        # optional name_filter: only consider this sample if any frame name equals the filter
        # or any file path contains the filter.
        if name_filter:
            name_match = any((n == name_filter) for n in names)
            file_match = any((name_filter in f) for f in files if f)
            if not (name_match or file_match):
                continue

        # For each package that appears in this sample, build a readable stack
        # (respecting name_filter) and use that as the deduplication key.
        for k in pkg_targets:
            if k not in joined:
                continue

            # Reconstruct filtered stack lines for this sample
            stack_lines: list[str] = []
            for fi_idx in idxs:
                if fi_idx is None:
                    continue
                try:
                    fii = int(fi_idx)
                except Exception:
                    continue
                if not (0 <= fii < len(frames)):
                    continue
                fi = frames[fii]
                name = fi.get("name", "") or ""
                file = fi.get("file", "") or ""
                line = fi.get("line")
                if name_filter:
                    if not (name == name_filter or (file and name_filter in file)):
                        continue
                if file:
                    if line is not None:
                        stack_lines.append(f"{name} ({file}:{line})")
                    else:
                        stack_lines.append(f"{name} ({file})")
                else:
                    stack_lines.append(f"{name}")

            # Use tuple of lines as a dedup key; skip empty stacks
            if not stack_lines:
                continue

            # If a filter_uniq substring is provided, dedupe based on only the
            # subset of lines that include that substring. If no such lines
            # exist in this stack, fall back to the full stack_lines so we
            # don't inadvertently collapse everything to the same key.
            if filter_uniq:
                filtered = [ln for ln in stack_lines if filter_uniq in ln]
                key = tuple(filtered) if filtered else tuple(stack_lines)
            else:
                key = tuple(stack_lines)

            # increment total and per-key counts so we can compute percentages
            pkg_total_counts[k] += 1
            pkg_key_counts[k][key] += 1

            # Add to examples list only the first time we see the key and if
            # we haven't reached max_examples yet.
            if key not in pkg_seen[k]:
                pkg_seen[k].add(key)
                if len(pkg_examples[k]) < max_examples:
                    pkg_examples[k].append(stack_lines)

    # print summaries for packages that have examples
    for k, examples in pkg_examples.items():
        if not examples:
            continue
        print()
        print(f"Sample stacks for package: {k}")

        # store rows without an id; we'll enumerate (1-indexed) after sorting
        # rows layout: (pct_pkg_val, pct_total_str, pct_pkg_str, stack_lines)
        rows: list[tuple[float, str, str, list[str]]] = []
        for ex in examples:
            # compute the same key used for dedup (respecting filter_uniq)
            if filter_uniq:
                filtered = [ln for ln in ex if filter_uniq in ln]
                key = tuple(filtered) if filtered else tuple(ex)
            else:
                key = tuple(ex)

            total = pkg_total_counts.get(k, 0) or 0
            key_count = pkg_key_counts.get(k, {}).get(key, 0)
            # pct for this key relative to package
            pct_pkg = 100.0 * key_count / total if total else 0.0
            pct_pkg_str = f"{pct_pkg:.1f}%"

            # pct for this key relative to all samples
            pct_total = 100.0 * key_count / overall_total if overall_total else 0.0

            if pct_total < 0.01 and pct_total > 0:
                pct_total_str = "< 0.01%"
            else:
                pct_total_str = f"{pct_total:.2f}%"

            # `rows` stores (pct_pkg_val, pct_total_str, pct_pkg_str, stack_lines)
            rows.append((pct_pkg, pct_total_str, pct_pkg_str, ex if ex else []))

        # Sort rows by pct_pkg descending (numeric)
        rows.sort(key=lambda r: r[0], reverse=True)

        # Compute content widths; id column is '#' and its width depends on
        # number of rows (max digits in row number)
        num_rows = len(rows)
        id_col_w = max(len("#"), len(str(num_rows)) if num_rows else 1)
        pct_total_col_w = max(len("pct total"), max((len(pct) for _, pct, _pctpkg, _ in rows), default=1))
        pct_pkg_col_w = max(len("pct package"), max((len(pct) for _, _pct, pct, _ in rows), default=1))

        stack_content_w = 0
        for _pct_pkg_val, _pct_total, _pct_pkg, stack_lines in rows:
            for ln in stack_lines:
                stack_content_w = max(stack_content_w, len(ln))

        # Build header cells (with single-space padding either side)
        id_cell = " " + "#".ljust(id_col_w) + " "
        pct_total_cell = " " + "pct total".ljust(pct_total_col_w) + " "
        pct_cell = " " + "pct package".ljust(pct_pkg_col_w) + " "
        # add one extra pad so the separator line is slightly longer (matches desired layout)
        stack_cell = " " + "stack".ljust(stack_content_w + 4) + " "
        header_line = "|" + id_cell + "|" + pct_total_cell + "|" + pct_cell + "|" + stack_cell + "|"
        sep_line = (
            "|"
            + ("-" * len(id_cell))
            + "|"
            + ("-" * len(pct_total_cell))
            + "|"
            + ("-" * len(pct_cell))
            + "|"
            + ("-" * len(stack_cell))
            + "|"
        )
        print(header_line)
        print(sep_line)

        # print rows: enumerate to produce 1-indexed '#' ids per package
        for idx, (_pct_pkg_val, pct_total_str, pct_pkg_str, stack_lines) in enumerate(rows, start=1):
            id_str = str(idx)
            if stack_lines:
                first = stack_lines[0]
                print(
                    f"| {id_str.ljust(id_col_w)} | {pct_total_str.ljust(pct_total_col_w)} | "
                    f"{pct_pkg_str.ljust(pct_pkg_col_w)} | {first.ljust(stack_content_w + 4)} |"
                )
                for ln in stack_lines[1:]:
                    print(
                        f"| {''.ljust(id_col_w)} | {''.ljust(pct_total_col_w)} | "
                        f"{''.ljust(pct_pkg_col_w)} | {ln.ljust(stack_content_w + 4)} |"
                    )
            else:
                print(
                    f"| {id_str.ljust(id_col_w)} | {''.ljust(pct_total_col_w)} | "
                    f"{''.ljust(pct_pkg_col_w)} | {''.ljust(stack_content_w + 4)} |"
                )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse a speedscope JSON file and print top imports by sample count.",
        epilog=(
            "Provide one or more package substrings to search for. Example:\n"
            "  python speedscope_top_packages.py /path/to/speedscope.json -p transformer_lens lightning sae_lens\n\n"
            "When --sample-stacks is used, the script prints a per-package table of example\n"
            "stacks with columns: '#', 'pct total', 'pct package', and 'stack'.\n\n"
            "  - '#' row number for the presented stack within that package (after\n"
            "    sorting by 'pct package' descending).\n"
            "  - 'pct total' is the percentage of overall samples represented by the stack\n"
            "    (formatted to two decimals; very small values are shown as '< 0.01%').\n"
            "  - 'pct package' is the percentage of that package's matching samples represented\n"
            "    by the stack (formatted to one decimal place).\n\n"
            "Use --filter-uniq to deduplicate stacks by a substring present in stack frame lines\n"
            "(e.g. a filename) so that only the matching lines are used when deciding uniqueness.\n"
        ),
    )
    p.add_argument("file", type=Path, help="Path to speedscope-format JSON file.")
    p.add_argument(
        "-p",
        "--packages",
        nargs="+",
        metavar="PKG",
        required=True,
        help="Space-separated list of package name substrings to search for. This argument is required.",
    )
    p.add_argument(
        "--sample-stacks",
        action="store_true",
        help=(
            "If set, also print example stacks for matching packages (appended to the report). "
            "The printed table contains columns '#', 'pct total', 'pct package', and 'stack'."
        ),
    )
    p.add_argument(
        "--name-filter",
        type=str,
        default=None,
        help="Optional substring to filter samples by frame name/file when sampling stacks (e.g. '<module>').",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum number of example stacks to print per package.",
    )
    p.add_argument(
        "--filter-uniq",
        type=str,
        default=None,
        help=(
            "Optional substring used when deduplicating stacks: only lines whose "
            "file path/name contains this substring are considered for uniqueness. "
            "If no lines in a stack match the substring, the full stack is used as the "
            "deduplication key (to avoid collapsing all stacks)."
        ),
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    pkgs = args.packages
    parse_speedscope(args.file, pkgs)
    if getattr(args, "sample_stacks", False):
        sample_stacks(
            args.file,
            pkgs,
            name_filter=args.name_filter,
            max_examples=args.max_examples,
            filter_uniq=getattr(args, "filter_uniq", None),
        )
