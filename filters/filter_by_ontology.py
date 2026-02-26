#!/usr/bin/env python3
"""
Filter papers using an ontology-based approach: require typed term co-occurrence.

- Triples: extracted from full paper text (all sections + tables).
- Negative terms: checked only in title/abstract/intro.

A paper passes if it contains "strong" triples such as:
  - (polymer, synthesizedBy, process)  e.g. RAFT, ATRP, ring-opening polymerization
  - (polymer, hasArchitecture, arch)   e.g. linear, branched, crosslinked
  - (polymer, hasProperty, property) with a numeric value  e.g. Tg 45 °C, Mw 50 kDa

Usage:
  python -m filters.filter_by_ontology corpus/2019/papers_passing -o passing.txt -f failures.txt
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from config import PROJECT_ROOT, PURPLE_BOOK_TERMS_FILE
from parsers.rsc_html_parser import RSCSectionParser

# Negative terms (poly* false positives)
NEGATIVE_TERMS = [
    "polynomial", "polygenic", "polysemy", "polyglot", "polycrystalline",
    "polytypism", "polyphase flow", "polymerase", "polyneuropathy",
    "polyhedral", "polytope",
]

# Typed lexicons - populated from Purple Book + known terms
POLYMERS: set[str] = set()
MONOMERS: set[str] = set()
PROPERTIES: set[str] = set()
PROCESSES: set[str] = set()
ARCH: set[str] = set()
METHODS: set[str] = set()

# Regex for property + numeric value (Tg 45 °C, Mw 50 kDa, etc.)
NUM_UNIT = re.compile(
    r"(?i)\b\d+(\.\d+)?\s*(°c|°C|K|GPa|MPa|kDa|Da|kg/mol|g/mol|Pa·s|Pa\s*s|wt%|mol%)\b"
)


def _load_lexicons():
    """Populate lexicons from Purple Book terms + known chemistry terms."""
    # Known chemistry terms not in Purple Book
    PROCESSES.update({
        "RAFT", "ATRP", "NMP", "ROMP", "ROP", "FRP", "free radical polymerization",
        "ring-opening polymerization", "ring opening polymerization",
        "chain polymerization", "condensation polymerization", "polycondensation",
        "living polymerization", "radical polymerization", "anionic polymerization",
        "cationic polymerization", "copolymerization", "homopolymerization",
        "step-growth", "step growth", "addition polymerization",
    })
    PROPERTIES.update({
        "Tg", "T_m", "Tm", "glass transition", "glass transition temperature",
        "melting temperature", "melting point", "crystallinity", "degree of crystallinity",
        "Mw", "Mn", "molecular weight", "molar mass", "polydispersity", "Đ", "D",
        "Young's modulus", "tensile strength", "viscosity", "melt viscosity",
        "degree of polymerization", "DP", "chain length",
    })
    ARCH.update({
        "linear", "branched", "crosslinked", "cross-linked", "star", "comb",
        "block", "graft", "dendritic", "hyperbranched", "network", "cyclic", 
        "brush", "stereoregular", 'atactic', 'isotactic', 'syndiotactic',
        "random","gradient", "periodic", "tapered","blocky", "correlated",
    })
    METHODS.update({
        "DSC", "DMA", "DMTA", "GPC", "SEC", "NMR", "FTIR", "IR", "XRD", "SAXS",
        "WAXS", "TEM", "SEM", "AFM", "TGA", "rheology",
    })
    POLYMERS.update({
        "polymer", "polymers", "macromolecule", "macromolecules", "copolymer",
        "copolymers", "homopolymer", "homopolymers", "oligomer", "oligomers",
    })
    MONOMERS.update({"monomer", "monomers"})

    # Load Purple Book terms and categorize by heuristics
    if PURPLE_BOOK_TERMS_FILE.exists():
        try:
            with open(PURPLE_BOOK_TERMS_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
        for item in (data if isinstance(data, list) else []):
            if not isinstance(item, dict) or "term" not in item:
                continue
            t = item["term"].strip()
            if not t or len(t) < 2:
                continue
            tl = t.lower()
            if tl in ("contents", "preamble", "terminology"):
                continue
            if "polymer" in tl or "macromolecule" in tl or "copolymer" in tl or "oligomer" in tl:
                if "monomer" in tl and "macromonomer" not in tl:
                    MONOMERS.add(t)
                else:
                    POLYMERS.add(t)
            elif "polymerization" in tl or "polymerisation" in tl or "copolymerization" in tl:
                PROCESSES.add(t)
            elif "linear" in tl or "branched" in tl or "crosslink" in tl or "star" in tl or "comb" in tl or "graft" in tl or "block" in tl:
                ARCH.add(t)
            elif "crystallinity" in tl or "transition" in tl or "temperature" in tl or "modulus" in tl or "molecular weight" in tl or "degree of" in tl:
                PROPERTIES.add(t)
            elif "DSC" in t or "DMA" in t or "GPC" in t or "NMR" in t or "X-ray" in tl or "calorimetry" in tl:
                METHODS.add(t)


def _split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _regex_poly_patterns(sentence: str) -> list[str]:
    """Find polymer mentions via regex: poly(X), polystyrene, polyvinyl, etc."""
    hits = []
    s = sentence
    for m in re.finditer(r"\bpoly\s*\(\s*([^)]+)\)", s, re.I):
        hits.append(f"poly({m.group(1).strip()})")
    for m in re.finditer(r"\b(poly(?:vinyl|styrene|ethylene|propylene|ester|amide|ether|urethane|imide|saccharide))\b", s, re.I):
        hits.append(m.group(1))
    return hits


def find_mentions(sentence: str, lexicon: set[str]) -> list[str]:
    """Find whole-word mentions of lexicon terms in sentence."""
    s = sentence.lower()
    hits = []
    for term in lexicon:
        if re.search(rf"\b{re.escape(term.lower())}\b", s):
            hits.append(term)
    return hits


def validate_triples(text: str) -> list[tuple[str, str, str, bool]]:
    """Extract ontology triples from text."""
    triples = []
    for sent in _split_into_sentences(text):
        polys = find_mentions(sent, POLYMERS) + _regex_poly_patterns(sent)
        props = find_mentions(sent, PROPERTIES)
        procs = find_mentions(sent, PROCESSES)
        archs = find_mentions(sent, ARCH)
        has_num = bool(NUM_UNIT.search(sent))

        for p in polys:
            for x in props:
                triples.append((p, "hasProperty", x, has_num))
            for x in procs:
                triples.append((p, "synthesizedBy", x, True))
            for x in archs:
                triples.append((p, "hasArchitecture", x, True))

    return triples


def _build_title_abstract_intro(meta: dict, sections: dict) -> str:
    """Build text from title, abstract, and introduction only (for negative term check)."""
    parts = [meta.get("title", "")]
    parts.append(sections.get("Abstract", ""))
    parts.append(sections.get("Introduction", "") or sections.get("1. Introduction", ""))
    return " ".join(p for p in parts if p and p.strip())


def _build_full_text(meta: dict, sections: dict, tables: list) -> str:
    """Build full paper text from title, all sections, and tables (for triple extraction)."""
    parts = [meta.get("title", "")]
    for section_name, section_text in sections.items():
        if section_text and section_text.strip():
            parts.append(section_text)
    for t in tables:
        content = t.get("content", "").strip()
        if content:
            parts.append(content)
    return " ".join(parts)


def is_polymer_paper(title_abstract_intro: str, full_text: str) -> tuple[bool, str]:
    """
    Determine if paper is about polymers.
    - Negative terms: checked only in title/abstract/intro.
    - Triples: extracted from full text (all sections + tables).
    """
    # Negative terms only in title/abstract/intro
    title_abstract_intro_lower = title_abstract_intro.lower()
    for neg in NEGATIVE_TERMS:
        if re.search(rf"\b{re.escape(neg)}\b", title_abstract_intro_lower):
            return False, f"negative term '{neg}' in abstract/title/intro"

    # Triples from full text
    triples = validate_triples(full_text)
    if not triples:
        return False, "no ontology triples found"

    for s, p, o, strong in triples:
        if p in ("synthesizedBy", "hasArchitecture") and strong:
            return True, f"strong triple: ({s}, {p}, {o})"
        if p == "hasProperty" and strong:
            return True, f"strong triple: ({s}, hasProperty, {o}) with value"

    return False, f"only weak triples ({len(triples)} found, need strong evidence)"


def filter_papers(
    folder: Path,
    output_file: Path | None = None,
    failures_file: Path | None = None,
    copy_to: Path | None = None,
    verbose: bool = True,
) -> list[Path]:
    """Filter papers in folder using ontology-based criteria."""
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
        title_abstract_intro = _build_title_abstract_intro(meta, sections)
        full_text = _build_full_text(meta, sections, tables)

        ok, reason = is_polymer_paper(title_abstract_intro, full_text)
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
            print(f"Copied {len(passing)} papers to {copy_to}", file=sys.stderr)

    return passing


def main():
    _load_lexicons()

    default_folder = PROJECT_ROOT / "corpus" / "2019" / "papers_passing"
    parser = argparse.ArgumentParser(
        description="Filter papers using ontology-based term co-occurrence"
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
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    passing = filter_papers(
        folder,
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
