#!/usr/bin/env python3
"""
Aggregate extracted JSON into a pandas DataFrame.

Each row = one property from one composition from one paper.
Paper and composition metadata are included as columns.

Usage:
  python aggregate_extracted.py extracted.json -o extracted.csv
  python aggregate_extracted.py extracted.json -o extracted.parquet
  python aggregate_extracted.py extracted.json  # print to stdout
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_extracted(path: Path) -> list[dict]:
    """Load extracted JSON (list of paper results)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def aggregate_to_dataframe(data: list[dict]) -> pd.DataFrame:
    """
    Flatten nested structure: one row per property.
    Columns: paper metadata + composition + property fields.
    """
    rows = []
    for paper in data:
        paper_meta = {
            "doi": paper.get("doi"),
            "title": paper.get("title"),
            "source_file": paper.get("source_file"),
            "subfield": paper.get("subfield"),
            "time_spent_seconds": paper.get("time_spent_seconds"),
        }
        for comp in paper.get("compositions", []):
            comp_meta = {
                "composition": comp.get("composition"),
                "composition_standard": comp.get("composition_standard"),
                "composition_abbreviations_resolved": comp.get("composition_abbreviations_resolved"),
                "processing_conditions": comp.get("processing_conditions"),
            }
            for prop in comp.get("properties_of_composition", []):
                row = {
                    **paper_meta,
                    **comp_meta,
                    "property_name": prop.get("property_name"),
                    "property_symbol": prop.get("property_symbol"),
                    "property_name_original": prop.get("property_name_original"),
                    "value": prop.get("value"),
                    "value_numeric": prop.get("value_numeric"),
                    "value_numeric_si": prop.get("value_numeric_si"),
                    "unit": prop.get("unit"),
                    "unit_si": prop.get("unit_si"),
                    "value_type": prop.get("value_type"),
                    "value_error": prop.get("value_error"),
                    "measurement_condition": prop.get("measurement_condition"),
                    "additional_information": prop.get("additional_information"),
                    "confidence": prop.get("confidence"),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate extracted JSON into a pandas DataFrame"
    )
    parser.add_argument(
        "input",
        type=Path,
        default=Path("extracted.json"),
        nargs="?",
        help="Input JSON file (default: extracted.json)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file (CSV or Parquet). If omitted, print to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        help="Output format (inferred from -o extension if not set)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    data = load_extracted(args.input)
    df = aggregate_to_dataframe(data)

    if args.output:
        fmt = args.format or args.output.suffix.lstrip(".").lower()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "parquet":
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"Wrote {len(df)} rows to {args.output}", file=sys.stderr)
    else:
        print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
