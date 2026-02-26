#!/usr/bin/env python3
"""
Extract polymer names, properties, and processes from papers.

KnowMat-style pipeline with TrustCall + Pydantic schema:
  1. Parse HTML (RSC parser)
  2. Subfield classifier (Ollama deepseek-r1:8b) - keep only experimental
  3. Extraction (OpenAI gpt-5.2 via TrustCall) - structured data with full property encoding
  4. Output: JSON only (compositions, characterisation, properties_of_composition)

Schema: KnowMat-style with value_numeric, measurement_condition, additional_information, property_symbol.

Usage:
  python extract_polymer_properties.py corpus/2019/papers_all_filters_passing -o extracted.json
  python extract_polymer_properties.py papers_folder --limit 10 --skip-subfield
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from config import DEFAULT_EXTRACTION_OUTPUT, DEFAULT_PAPERS_FOLDER, PROJECT_ROOT
from extraction.extractors import extract_with_trustcall
try:
    from extraction.cde2_integration import (
        extract_tables_cde2,
        extract_properties_rulebased,
        merge_uncertainty_and_rulebased,
        is_available as cde2_available,
    )
except (ImportError, Exception):
    def cde2_available():
        return False
    def extract_tables_cde2(_):
        return []
    def extract_properties_rulebased(_):
        return []
    def merge_uncertainty_and_rulebased(llm_props, _cde2_props, _comp):
        return llm_props
from extraction.post_processing import post_process_compositions
from extraction.prompt_generator import generate_system_prompt, generate_user_prompt
from parsers.rsc_html_parser import RSCSectionParser

load_dotenv(PROJECT_ROOT / ".env")

SUBFIELD_PROMPT = """Classify this polymer/materials science paper. Reply with ONLY one word.

Options: experimental, computational, simulation, machine_learning, hybrid, other

- experimental: reports synthesis, characterization, measured properties
- computational: DFT, MD, theory, modelling (no lab work)
- simulation: molecular dynamics, Monte Carlo, etc.
- machine_learning: ML models for property prediction
- hybrid: combines experimental + computational
- other: review, methodology, not materials-focused

Paper:
Title: {title}
Abstract: {abstract}

Reply with exactly one word:"""


def _table_content_to_markdown(content: str) -> str:
    """
    Convert table content (rows separated by newlines, cells by ' | ') to markdown.
    First row is treated as header.
    """
    if not content or not content.strip():
        return content
    rows = [r.strip() for r in content.strip().split("\n") if r.strip()]
    if not rows:
        return content
    # Split each row by ' | ' to get cells
    parsed = [r.split(" | ") for r in rows]
    if not parsed:
        return content
    ncols = max(len(r) for r in parsed)
    # Pad rows to same length
    for r in parsed:
        r.extend([""] * (ncols - len(r)))
    # Build markdown: header, separator, body
    header = "| " + " | ".join(parsed[0]) + " |"
    sep = "| " + " | ".join(["---"] * ncols) + " |"
    body = "\n".join("| " + " | ".join(cell for cell in r) + " |" for r in parsed[1:])
    return header + "\n" + sep + "\n" + body if body else header + "\n" + sep


def get_paper_text(
    parser_result: dict,
    include_tables: bool = True,
    table_aware: bool = True,
    html_path: Path | None = None,
    use_cde2_tables: bool = True,
) -> str:
    """Extract full paper text for LLM: all sections and tables (markdown when table_aware)."""
    meta = parser_result.get("meta", {})
    sections = parser_result.get("sections", {})
    tables = parser_result.get("tables", [])
    if use_cde2_tables and html_path and cde2_available():
        cde2_tables = extract_tables_cde2(html_path)
        if cde2_tables:
            tables = [{"caption": t.get("caption", ""), "content": t.get("content", "")} for t in cde2_tables]
    title = meta.get("title", "")
    parts = [f"Title: {title}"]
    for section_name, section_text in sections.items():
        if section_text and section_text.strip():
            parts.append(f"{section_name}:\n{section_text}")
    if include_tables and tables:
        table_parts = []
        for i, t in enumerate(tables):
            caption = t.get("caption", "").strip()
            content = t.get("content", "").strip()
            if caption or content:
                tbl = f"Table {i + 1}"
                if caption:
                    tbl += f" ({caption})"
                if content:
                    formatted = _table_content_to_markdown(content) if table_aware else content
                    tbl += f":\n{formatted}"
                else:
                    tbl += f": {caption}"
                table_parts.append(tbl)
        if table_parts:
            parts.append("Tables (rows/columns structure):\n" + "\n\n".join(table_parts))
    return "\n\n".join(p for p in parts if (p.split(":", 1)[-1] if ":" in p else p).strip())


def classify_subfield(title: str, abstract: str, model: str = "deepseek-r1:8b") -> str:
    """Use Ollama to classify paper subfield. Returns experimental/computational/etc."""
    try:
        from ollama import chat
    except ImportError:
        return "experimental"

    prompt = SUBFIELD_PROMPT.format(title=title[:500], abstract=abstract[:2000])
    try:
        response = chat(model=model, messages=[{"role": "user", "content": prompt}])
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
        else:
            content = getattr(getattr(response, "message", None), "content", "") or ""
        label = content.strip().lower().split()[0] if content else "other"
        for opt in ("experimental", "computational", "simulation", "machine_learning", "hybrid", "other"):
            if opt in label or label in opt:
                return opt
        return "other"
    except Exception:
        return "experimental"


def load_property_filter(properties_file: Path | None, properties_list: list[str] | None) -> list[str] | None:
    """Load property names from file or list. Returns None if neither provided."""
    if properties_list:
        return [p.strip() for p in properties_list if p.strip()]
    if properties_file and properties_file.exists():
        text = properties_file.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            return json.loads(text)
        return [line.strip() for line in text.splitlines() if line.strip()]
    return None


def filter_compositions_by_properties(data: dict, property_filter: list[str]) -> dict:
    """Post-filter: keep only requested properties in each composition."""
    if not property_filter:
        return data
    allowed = {p.lower() for p in property_filter}
    for comp in data.get("compositions", []):
        props = comp.get("properties_of_composition", [])
        filtered = [
            p for p in props
            if (p.get("property_name", "").lower() in allowed
                or (p.get("property_symbol") and p["property_symbol"].lower() in allowed))
        ]
        comp["properties_of_composition"] = filtered
    return data


def extract_with_trustcall_wrapper(
    text: str,
    subfield: str,
    model: str = "gpt-5.2",
    property_filter: list[str] | None = None,
) -> dict:
    """Build KnowMat-style prompt and run extraction (TrustCall or Ollama JSON fallback)."""
    from extraction.extractors import _is_ollama_json_fallback
    system_prompt = generate_system_prompt(sub_field=subfield, property_filter=property_filter)
    user_prompt = generate_user_prompt(text)
    if model.lower().startswith("ollama:") and _is_ollama_json_fallback(model):
        suffix = "Respond with valid JSON only, no other text."
    else:
        suffix = "Provide your response using the tool."
    full_prompt = system_prompt + "\n\n" + user_prompt + "\n\n" + suffix
    data = extract_with_trustcall(full_prompt, model=model)
    if property_filter:
        data = filter_compositions_by_properties(data, property_filter)
    return data


def process_paper(
    path: Path,
    extraction_model: str,
    subfield_model: str,
    skip_subfield: bool,
    property_filter: list[str] | None = None,
    use_cde2_tables: bool = True,
    use_cde2_rulebased: bool = True,
) -> dict | None:
    """
    Process one paper. Returns extraction dict or None if skipped (non-experimental).
    """
    t0 = time.perf_counter()
    try:
        parser = RSCSectionParser(path, include_tables=True)
        result = parser.to_dict()
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"doi": path.stem, "error": str(e), "compositions": [], "skipped": False, "time_spent_seconds": round(elapsed, 2)}

    meta = result.get("meta", {})
    sections = result.get("sections", {})
    title = meta.get("title", "")
    abstract = sections.get("Abstract", "")
    doi = meta.get("doi", path.stem)

    subfield = "unknown"
    if not skip_subfield:
        subfield = classify_subfield(title, abstract, model=subfield_model)
        if subfield not in ("experimental", "hybrid"):
            return None

    # 1. CDE2 rule-based extraction first (when available and enabled)
    cde2_props = []
    if use_cde2_rulebased and cde2_available():
        cde2_props = extract_properties_rulebased(path)

    # 2. LLM extraction
    text = get_paper_text(result, html_path=path, use_cde2_tables=use_cde2_tables)
    data = extract_with_trustcall_wrapper(
        text, subfield=subfield, model=extraction_model, property_filter=property_filter
    )

    # 3. Merge CDE2 into LLM results (uncertainties, CDE2-only properties)
    if cde2_props:
        for comp in data.get("compositions", []):
            comp_name = comp.get("composition", "")
            props = comp.get("properties_of_composition", [])
            comp["properties_of_composition"] = merge_uncertainty_and_rulebased(
                props, cde2_props, comp_name
            )

    # 4. Post-process: always LLM alignment (term translation), validation, SI, confidence
    data = post_process_compositions(
        data, align_ontology=True, validate=True, add_confidence=True,
        add_si=True, add_uncertainty=True,
        use_llm_alignment=True,
        alignment_model=extraction_model,
    )

    elapsed = time.perf_counter() - t0
    data["doi"] = doi
    data["title"] = title
    data["source_file"] = path.name
    data["subfield"] = subfield
    data["time_spent_seconds"] = round(elapsed, 2)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Extract polymer data (TrustCall + KnowMat schema)"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(DEFAULT_PAPERS_FOLDER),
        help="Folder with papers",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_OUTPUT,
        help="Output JSON file",
    )
    parser.add_argument(
        "--properties-file",
        type=Path,
        help="JSON or text file with property names to extract (one per line or JSON list)",
    )
    parser.add_argument(
        "--properties-list",
        type=str,
        nargs="+",
        help="Property names to extract (e.g. Tg Mw Mn)",
    )
    parser.add_argument(
        "--extraction-model",
        default="gpt-5.2",
        help="Extraction model: gpt-4o, gpt-5.2 (OpenAI); claude-3-5-sonnet (Anthropic); ollama:llama3.1, ollama:deepseek-r1:8b (local). Default: gpt-5.2",
    )
    parser.add_argument(
        "--subfield-model",
        default="deepseek-r1:8b",
        help="Ollama model for subfield classification (default: deepseek-r1:8b)",
    )
    parser.add_argument(
        "--skip-subfield",
        action="store_true",
        help="Skip subfield filter, extract from all papers",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N papers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last run: load existing output, skip already-processed papers, append new results",
    )
    parser.add_argument(
        "--no-cde2",
        action="store_true",
        help="Disable all CDE2 (tables + rule-based). Use when CDE2 model (hf_bert_crf_tagger) cannot be downloaded. Falls back to RSC parser.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode",
    )
    args = parser.parse_args()

    property_filter = load_property_filter(args.properties_file, args.properties_list)
    extraction_model = args.extraction_model.lower()
    if extraction_model.startswith("ollama:"):
        pass  # No API key needed; Ollama runs locally
    elif extraction_model.startswith("claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Set ANTHROPIC_API_KEY for Claude models (e.g. in .env)", file=sys.stderr)
            sys.exit(1)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY for OpenAI models (e.g. in .env)", file=sys.stderr)
            sys.exit(1)

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    html_files = sorted(folder.glob("*.html"))
    if not html_files:
        print(f"No HTML files in {folder}", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        html_files = html_files[: args.limit]

    results = []
    verbose = not args.quiet
    already_processed: set[str] = set()
    if args.resume and args.output.exists():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for item in existing:
                    sf = item.get("source_file")
                    if sf:
                        already_processed.add(sf)
            if already_processed and verbose:
                print(f"Resuming: skipping {len(already_processed)} already-processed papers", file=sys.stderr)
        except (json.JSONDecodeError, OSError):
            pass

    html_files = [p for p in html_files if p.name not in already_processed]
    skipped = 0

    for i, path in enumerate(html_files):
        if verbose:
            print(f"  [{i+1}/{len(html_files)}] {path.name}...", file=sys.stderr)

        data = process_paper(
            path,
            extraction_model=args.extraction_model,
            subfield_model=args.subfield_model,
            skip_subfield=args.skip_subfield,
            property_filter=property_filter,
            use_cde2_tables=not args.no_cde2,
            use_cde2_rulebased=not args.no_cde2,
        )
        if data is None:
            skipped += 1
            continue
        results.append(data)

    if args.resume and args.output.exists() and already_processed:
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                results = existing + results
        except (json.JSONDecodeError, OSError):
            pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_compositions = sum(len(r.get("compositions", [])) for r in results)
    total_seconds = sum(r.get("time_spent_seconds", 0) for r in results)
    if verbose:
        print(f"\nExtracted {total_compositions} compositions from {len(results)} papers", file=sys.stderr)
        if not args.skip_subfield:
            print(f"Skipped {skipped} non-experimental papers", file=sys.stderr)
        print(f"Total time: {total_seconds:.1f} s ({total_seconds / 60:.1f} min)", file=sys.stderr)
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
