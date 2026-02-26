#!/usr/bin/env python3
"""
Filter RSC HTML papers to find those actually about polymers.

Papers were downloaded based on 'poly*' in abstract, but many are false positives
(e.g. polynomial, polycrystalline, polymerase). This script uses positive and
negative term lists to filter for polymer-related papers.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from config import DEFAULT_PAPERS_INPUT, PURPLE_BOOK_TERMS_FILE, PROJECT_ROOT
from parsers.rsc_html_parser import RSCSectionParser


# Terms that indicate the paper is NOT about polymers (poly* false positives)
NEGATIVE_TERMS = [
    "polynomial",
    "polygenic",
    "polysemy",
    "polyglot",
    "polycrystalline",
    "polytypism",
    "polyphase flow",
    "polymerase",
    "polyneuropathy",
    "polyhedral",
    "polytope",
]

# Terms that indicate the paper IS about polymers
POSITIVE_TERMS = [
    "polymer",
    "polymeric",
    "macromolecule",
    "macromolecular",
    "copolymer",
    "homopolymer",
    "polymerization",
    "polymerisation",
    "monomer",
    "oligomer",
    "chain length",
    "degree of polymerization",
    "molecular weight",
    "polydispersity",
    "melt",
    "amorphous",
    "semicrystalline",
]


def _load_purple_book_terms(path: Path | None = None) -> list[str]:
    """Load terms from Purple Book extraction JSON. Returns list of term strings."""
    path = path or PURPLE_BOOK_TERMS_FILE
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item["term"] for item in data if isinstance(item, dict) and "term" in item]
        return []
    except Exception:
        return []


def _get_positive_terms(include_purple_book: bool = True) -> list[str]:
    """Return positive terms, optionally merged with Purple Book terms."""
    terms = list(POSITIVE_TERMS)
    if include_purple_book:
        purple = _load_purple_book_terms()
        for t in purple:
            if t.lower() not in {x.lower() for x in terms}:
                terms.append(t)
    return terms


def _count_term_matches(text: str, terms: list[str]) -> int:
    """Count how many of the terms appear in text (case-insensitive, whole-word)."""
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for term in terms:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, text_lower):
            count += 1
    return count


def _get_term_occurrences(text: str, terms: list[str]) -> dict[str, int]:
    """Return dict of term -> occurrence count in text."""
    if not text:
        return {}
    text_lower = text.lower()
    result = {}
    for term in terms:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        matches = re.findall(pattern, text_lower)
        if matches:
            result[term] = len(matches)
    return result


def is_polymer_paper(
    parser_result: dict,
    positive_terms: list[str] | None = None,
    negative_terms: list[str] | None = None,
    require_abstract: bool = True,
    min_positive: int = 1,
    max_negative_in_abstract: int = 0,
) -> tuple[bool, str]:
    """
    Determine if a paper is about polymers based on term counts.

    Args:
        parser_result: Output from RSCSectionParser.to_dict()
        positive_terms: Terms indicating polymer topic
        negative_terms: Terms indicating non-polymer topic (false positives)
        require_abstract: If True, check abstract; else use full text
        min_positive: Minimum positive terms required
        max_negative_in_abstract: Max negative terms allowed in abstract (0 = none)

    Returns:
        (is_polymer, reason)
    """
    positive_terms = positive_terms or _get_positive_terms()
    negative_terms = negative_terms or NEGATIVE_TERMS

    sections = parser_result.get("sections", {})
    abstract = sections.get("Abstract", "")
    intro = sections.get("Introduction", "") or sections.get("1. Introduction", "")
    text_to_check = abstract + " " + intro if require_abstract else " ".join(sections.values())

    pos_matches = _count_term_matches(text_to_check, positive_terms)
    neg_matches = _count_term_matches(abstract or text_to_check, negative_terms)

    if pos_matches < min_positive:
        return False, f"only {pos_matches} positive term(s), need {min_positive}"

    if neg_matches > max_negative_in_abstract:
        return False, f"{neg_matches} negative term(s) in abstract"

    return True, f"{pos_matches} positive, {neg_matches} negative"


def filter_papers(
    folder: Path,
    output_file: Path | None = None,
    failures_file: Path | None = None,
    copy_to: Path | None = None,
    verbose: bool = True,
    include_purple_book: bool = True,
) -> list[Path]:
    """
    Filter papers in folder, return list of paths that pass.
    """
    html_files = sorted(folder.glob("*.html"))
    if not html_files:
        print(f"No HTML files found in {folder}", file=sys.stderr)
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text("", encoding="utf-8")
        return []

    passing = []
    failing = []

    for i, path in enumerate(html_files):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(html_files)}...", file=sys.stderr)

        try:
            parser = RSCSectionParser(path, include_tables=False)
            result = parser.to_dict()
        except Exception as e:
            if verbose:
                print(f"  Error parsing {path.name}: {e}", file=sys.stderr)
            failing.append((path, f"parse error: {e}"))
            continue

        pos_terms = _get_positive_terms(include_purple_book=include_purple_book)
        is_poly, reason = is_polymer_paper(result, positive_terms=pos_terms)
        if is_poly:
            passing.append(path)
        else:
            failing.append((path, reason))

    if verbose:
        print(f"\nPassed: {len(passing)} / {len(html_files)}", file=sys.stderr)
        print(f"Failed: {len(failing)}", file=sys.stderr)

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for p in passing:
                f.write(f"{p.name}\n")
        if verbose:
            print(f"Wrote list to {output_file}", file=sys.stderr)

    if failures_file:
        failures_file = Path(failures_file)
        failures_file.parent.mkdir(parents=True, exist_ok=True)
        with open(failures_file, "w") as f:
            for p, reason in failing:
                f.write(f"{p.name}\t{reason}\n")
        if verbose:
            print(f"Wrote failures to {failures_file}", file=sys.stderr)

    if copy_to:
        copy_to = Path(copy_to)
        copy_to.mkdir(parents=True, exist_ok=True)
        for p in passing:
            shutil.copy2(p, copy_to / p.name)
        if verbose:
            print(f"Copied {len(passing)} papers to {copy_to}", file=sys.stderr)

    return passing


def main():
    parser = argparse.ArgumentParser(
        description="Filter RSC papers to find those about polymers"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Folder containing RSC HTML papers (default: corpus/2019)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write list of passing papers to file",
    )
    parser.add_argument(
        "-f", "--failures",
        help="Write failing papers with reasons to file",
    )
    parser.add_argument(
        "--copy-to",
        help="Copy passing papers to this directory",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--no-purple-book",
        action="store_true",
        help="Do not merge Purple Book terms into positive term list",
    )
    args = parser.parse_args()

    folder = Path(args.folder) if args.folder else DEFAULT_PAPERS_INPUT
    if not folder.exists():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    passing = filter_papers(
        folder,
        output_file=args.output,
        failures_file=args.failures,
        copy_to=args.copy_to,
        verbose=not args.quiet,
        include_purple_book=not args.no_purple_book,
    )

    if not args.quiet:
        print("\nPassing papers:")
        for p in passing[:20]:
            print(f"  {p.name}")
        if len(passing) > 20:
            print(f"  ... and {len(passing) - 20} more")


if __name__ == "__main__":
    main()
