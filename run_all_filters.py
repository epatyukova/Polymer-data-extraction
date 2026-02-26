#!/usr/bin/env python3
"""
Run all filters and output papers that pass ALL of them.

Pipeline:
  1. filter_polymer_papers (term-based)
  2. filter_by_ontology (triple-based)
  3. filter_by_embedding (similarity to examples)
  4. Intersection -> papers_all_filters_passing/

Usage:
  python run_all_filters.py corpus/2019
  python run_all_filters.py corpus/2019 -o corpus/2019/papers_all_passing --copy
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_filter(module: str, folder: Path, output: Path, extra_args: list[str] | None = None) -> set[str]:
    """Run filter module, return set of passing filenames."""
    cmd = [sys.executable, "-m", module, str(folder), "-o", str(output), "-q"]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True, capture_output=True, cwd=PROJECT_ROOT)
    return {line.strip() for line in output.read_text().splitlines() if line.strip()}


def main():
    parser = argparse.ArgumentParser(description="Run all filters, output intersection")
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(PROJECT_ROOT / "corpus" / "2019" / "papers_passing"),
        help="Folder with RSC HTML papers",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output dir for papers passing all filters",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy passing papers to output dir",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=None,
        help="Filter by property terms (polymer, property, value triples). Uses polymer_synonyms.json. E.g. --properties 'glass transition' Tg Mw Mn",
    )
    parser.add_argument(
        "--properties-file",
        type=Path,
        default=None,
        help="File with property names (one per line). Overrides --properties.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding filter (faster)",
    )
    parser.add_argument(
        "--embedding-percentile",
        type=float,
        default=25,
        help="Keep top N%% for embedding (default: 25)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet

    # Auto-detect: if folder has no HTML but has papers/ or papers_passing/ subdir, use that
    if not list(folder.glob("*.html")):
        for sub in ("papers", "papers_passing", "papers_onto_passing"):
            candidate = folder / sub
            if candidate.exists() and list(candidate.glob("*.html")):
                if verbose:
                    print(f"No HTML in {folder}; using {candidate}\n", file=sys.stderr)
                folder = candidate
                break

    out_dir = args.output_dir or (folder.parent / "papers_all_filters_passing")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        t1 = tmp / "term.txt"
        t2 = tmp / "ontology.txt"
        t3 = tmp / "embedding.txt"

        if verbose:
            print("Running term filter...", file=sys.stderr)
        term_set = run_filter("filters.filter_polymer_papers", folder, t1)
        if verbose:
            print(f"  {len(term_set)} passed\n", file=sys.stderr)

        if verbose:
            print("Running ontology filter...", file=sys.stderr)
        onto_set = run_filter("filters.filter_by_ontology", folder, t2)
        if verbose:
            print(f"  {len(onto_set)} passed\n", file=sys.stderr)

        property_set = None
        if args.properties or (args.properties_file and args.properties_file.exists()):
            if verbose:
                print("Running property-terms filter...", file=sys.stderr)
            t_prop = tmp / "property.txt"
            extra = []
            if args.properties:
                extra = ["--properties"] + args.properties
            elif args.properties_file:
                extra = ["--properties-file", str(args.properties_file)]
            cmd = [sys.executable, "-m", "filters.filter_by_property_terms", str(folder), "-o", str(t_prop), "-q"] + extra
            subprocess.run(cmd, check=True, capture_output=True, cwd=PROJECT_ROOT)
            property_set = {line.strip() for line in t_prop.read_text().splitlines() if line.strip()}
            if verbose:
                print(f"  {len(property_set)} passed\n", file=sys.stderr)

        if args.skip_embedding:
            emb_set = term_set  # no extra filter
            if verbose:
                print("Skipping embedding filter.\n", file=sys.stderr)
        else:
            if verbose:
                print("Running embedding filter...", file=sys.stderr)
            subprocess.run([
                sys.executable, "-m", "filters.filter_by_embedding",
                "--papers", str(folder), "-o", str(t3), "--percentile", str(args.embedding_percentile), "-q"
            ], check=True, capture_output=True, cwd=PROJECT_ROOT)
            emb_set = {line.strip().split("\t")[0] for line in t3.read_text().splitlines() if line.strip()}
            if verbose:
                print(f"  {len(emb_set)} passed\n", file=sys.stderr)

    all_pass = term_set & onto_set & emb_set
    if property_set is not None:
        all_pass = all_pass & property_set
    if verbose:
        print(f"Intersection: {len(all_pass)} papers pass all filters", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    list_file = out_dir / "passing_papers.txt"
    with open(list_file, "w") as f:
        for name in sorted(all_pass):
            f.write(f"{name}\n")

    if args.copy:
        for name in all_pass:
            src = folder / name
            if src.exists():
                shutil.copy2(src, out_dir / name)
        if verbose:
            print(f"Copied to {out_dir}", file=sys.stderr)

    if verbose:
        print(f"\nExtract properties:\n  python extract_polymer_properties.py {out_dir} -o extracted.json --limit 10")


if __name__ == "__main__":
    main()
