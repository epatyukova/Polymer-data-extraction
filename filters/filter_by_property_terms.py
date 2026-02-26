#!/usr/bin/env python3
"""
Filter papers by whether they contain specific property terms (with synonyms)
and valid triples: (polymer name, property, property value).

Uses extraction/polymer_synonyms.json for synonym expansion.
A paper passes if it contains at least one (polymer, property, value) triple
where the property matches the requested terms or any of their synonyms.

Usage:
  python -m filters.filter_by_property_terms corpus/2019/papers -o passing.txt --properties "glass transition" Tg Mw Mn
  python -m filters.filter_by_property_terms corpus/2019/papers -o passing.txt --properties-file properties.txt
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from config import PROJECT_ROOT
from parsers.rsc_html_parser import RSCSectionParser

POLYMER_SYNONYMS_PATH = PROJECT_ROOT / "extraction" / "polymer_synonyms.json"

# Regex for property + numeric value (Tg 45 °C, Mw 50 kDa, glass transition 105 °C, etc.)
NUM_UNIT = re.compile(
    r"(?i)\b\d+(\.\d+)?\s*(°c|°C|K|GPa|MPa|kDa|Da|kg/mol|g/mol|Pa·s|Pa\s*s|wt%|mol%)\b"
)
# Also match numbers without units (e.g. "Tg of 105")
NUM_ONLY = re.compile(r"\b\d+(\.\d+)?\b")


def _load_polymer_synonyms(path: Path | None = None) -> dict:
    """Load polymer_synonyms.json. Returns dict of category -> {canonical: [synonyms]}."""
    path = path or POLYMER_SYNONYMS_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, dict) and k.startswith("_")}


def _build_property_search_terms(
    requested: list[str],
    synonyms_data: dict,
) -> set[str]:
    """
    Expand requested property terms using synonyms.
    Returns set of all terms to search for (canonical + synonyms).
    """
    # Flatten: collect all {canonical: [synonyms]} from all categories
    all_mappings: dict[str, list[str]] = {}
    for category, mappings in synonyms_data.items():
        if isinstance(mappings, dict):
            for canonical, syn_list in mappings.items():
                if isinstance(syn_list, list):
                    all_mappings[canonical] = [s for s in syn_list if isinstance(s, str)]

    # Build reverse: term -> canonical (first match wins to avoid Tg->TGA overwriting Tg->glass transition)
    term_to_canonical: dict[str, str] = {}
    for canonical, syns in all_mappings.items():
        if canonical.lower() not in term_to_canonical:
            term_to_canonical[canonical.lower()] = canonical
        for s in syns:
            sl = s.lower()
            if sl not in term_to_canonical:
                term_to_canonical[sl] = canonical

    # For each requested term, find its canonical and collect all terms in that group
    search_terms: set[str] = set()
    requested_lower = [r.strip().lower() for r in requested if r.strip()]

    for req in requested_lower:
        canonical = term_to_canonical.get(req)
        if canonical:
            group = {canonical} | set(all_mappings.get(canonical, []))
            search_terms.update(group)
        else:
            # Term not in synonyms - add as-is for exact match
            search_terms.add(req)

    return search_terms


def _split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _regex_poly_patterns(sentence: str) -> list[str]:
    """Find polymer mentions via regex."""
    hits = []
    for m in re.finditer(r"\bpoly\s*\(\s*([^)]+)\)", sentence, re.I):
        hits.append(f"poly({m.group(1).strip()})")
    for m in re.finditer(
        r"\b(poly(?:vinyl|styrene|ethylene|propylene|ester|amide|ether|urethane|imide|saccharide))\b",
        sentence,
        re.I,
    ):
        hits.append(m.group(1))
    return hits


# Polymer lexicon for triple detection
POLYMER_TERMS = {
    "polymer", "polymers", "macromolecule", "macromolecules", "copolymer",
    "copolymers", "homopolymer", "homopolymers", "oligomer", "oligomers",
}


def _find_whole_word(sentence: str, terms: set[str]) -> list[str]:
    """Find whole-word mentions of terms in sentence (case-insensitive)."""
    s_lower = sentence.lower()
    hits = []
    for term in terms:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, s_lower):
            hits.append(term)
    return hits


def _has_valid_triple(
    sentence: str,
    property_terms: set[str],
) -> bool:
    """
    Check if sentence contains (polymer, property, value) triple.
    - Polymer mention (poly(X), polymer, copolymer, etc.)
    - Property term (or synonym)
    - Numeric value (with optional unit)
    """
    polys = _find_whole_word(sentence, POLYMER_TERMS) + _regex_poly_patterns(sentence)
    if not polys:
        return False

    props = _find_whole_word(sentence, property_terms)
    if not props:
        return False

    has_num = bool(NUM_UNIT.search(sentence)) or bool(NUM_ONLY.search(sentence))
    if not has_num:
        return False

    return True


def _build_full_text(meta: dict, sections: dict, tables: list) -> str:
    """Build full paper text from title, all sections, and tables."""
    parts = [meta.get("title", "")]
    for section_name, section_text in sections.items():
        if section_text and section_text.strip():
            parts.append(section_text)
    for t in tables:
        content = t.get("content", "").strip()
        if content:
            parts.append(content)
    return " ".join(parts)


def is_paper_with_property_triples(
    full_text: str,
    property_terms: set[str],
) -> tuple[bool, str]:
    """
    Determine if paper contains at least one (polymer, property, value) triple
    where property matches one of the requested terms or synonyms.
    """
    for sent in _split_into_sentences(full_text):
        if _has_valid_triple(sent, property_terms):
            return True, f"found (polymer, property, value) triple"
    return False, "no matching (polymer, property, value) triples"


def filter_papers(
    folder: Path,
    property_terms: list[str],
    synonyms_path: Path | None = None,
    output_file: Path | None = None,
    failures_file: Path | None = None,
    copy_to: Path | None = None,
    verbose: bool = True,
) -> list[Path]:
    """Filter papers that contain (polymer, property, value) triples for requested properties."""
    synonyms_data = _load_polymer_synonyms(synonyms_path)
    search_terms = _build_property_search_terms(property_terms, synonyms_data)

    if not search_terms:
        search_terms = {t.lower() for t in property_terms if t.strip()}

    if verbose:
        print(f"Property search terms ({len(search_terms)}): {sorted(search_terms)[:15]}...", file=sys.stderr)

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
            parser = RSCSectionParser(path, include_tables=True)
            result = parser.to_dict()
        except Exception as e:
            if verbose:
                print(f"  Error parsing {path.name}: {e}", file=sys.stderr)
            failing.append((path, f"parse error: {e}"))
            continue

        meta = result.get("meta", {})
        sections = result.get("sections", {})
        tables = result.get("tables", [])
        full_text = _build_full_text(meta, sections, tables)

        ok, reason = is_paper_with_property_triples(full_text, search_terms)
        if ok:
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
            print(f"Copied to {copy_to}", file=sys.stderr)

    return passing


def main():
    default_folder = PROJECT_ROOT / "corpus" / "2019" / "papers_passing"
    parser = argparse.ArgumentParser(
        description="Filter papers by (polymer, property, value) triples using property synonyms"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(default_folder),
        help="Folder containing RSC HTML papers",
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
        "--properties",
        nargs="+",
        default=[],
        help="Property terms to filter for (e.g. 'glass transition' Tg Mw Mn). Synonyms from polymer_synonyms.json are expanded.",
    )
    parser.add_argument(
        "--properties-file",
        type=Path,
        help="File with property names (one per line or JSON list). Overrides --properties.",
    )
    parser.add_argument(
        "--synonyms",
        type=Path,
        default=POLYMER_SYNONYMS_PATH,
        help=f"Path to polymer synonyms JSON (default: {POLYMER_SYNONYMS_PATH.name})",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    property_terms = args.properties
    if args.properties_file and args.properties_file.exists():
        text = args.properties_file.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            property_terms = json.loads(text)
        else:
            property_terms = [line.strip() for line in text.splitlines() if line.strip()]

    if not property_terms:
        print("Error: specify --properties or --properties-file", file=sys.stderr)
        sys.exit(1)

    passing = filter_papers(
        folder,
        property_terms=property_terms,
        synonyms_path=args.synonyms,
        output_file=args.output,
        failures_file=args.failures,
        copy_to=args.copy_to,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\nPassing papers:")
        for p in passing[:20]:
            print(f"  {p.name}")
        if len(passing) > 20:
            print(f"  ... and {len(passing) - 20} more")


if __name__ == "__main__":
    main()
