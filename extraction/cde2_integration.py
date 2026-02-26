"""
ChemDataExtractor2 integration (optional).

Provides:
- Table extraction via CDE2/TableDataExtractor
- Rule-based property pre-extraction (Tg, Tm, Mw, Mn, etc.)
- Uncertainty extraction from CDE2 records

All functions gracefully degrade when chemdataextractor2 is not installed.
"""

from pathlib import Path
from typing import Any

_CDE2_AVAILABLE = False
try:
    from chemdataextractor import Document
    from chemdataextractor.model import GlassTransition, MeltingPoint, Compound

    _CDE2_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Return True if ChemDataExtractor2 is installed."""
    return _CDE2_AVAILABLE


def extract_tables_cde2(html_path: str | Path) -> list[dict]:
    """
    Extract tables from HTML using ChemDataExtractor2 (TableDataExtractor).
    Returns list of {"caption": str, "content": str, "rows": list[list[str]]}.
    Falls back to empty list if CDE2 not available.
    """
    if not _CDE2_AVAILABLE:
        return []

    try:
        path = Path(html_path)
        with open(path, "rb") as f:
            doc = Document.from_file(f, fname=path.name)
    except Exception:
        return []

    tables_out = []
    for table_el in doc.tables:
        caption = ""
        if hasattr(table_el, "caption") and table_el.caption:
            caption = getattr(table_el.caption, "text", "") or str(table_el.caption)

        rows = []
        if hasattr(table_el, "tde_table") and table_el.tde_table is not None:
            tde = table_el.tde_table
            # Get table data from TableDataExtractor
            if hasattr(tde, "category_table") and tde.category_table:
                for row in tde.category_table:
                    cells = [str(c).strip() if c else "" for c in row]
                    rows.append(cells)
            elif hasattr(tde, "data") and tde.data:
                rows = [[str(c) for c in row] for row in tde.data]

        content = ""
        if rows:
            content = "\n".join(" | ".join(cell for cell in row) for row in rows)

        tables_out.append({"caption": caption, "content": content, "rows": rows})
    return tables_out


def extract_properties_rulebased(html_path: str | Path) -> list[dict]:
    """
    Rule-based extraction of Tg, Tm, and other polymer properties via CDE2.
    Returns list of {"property_name": str, "value": str, "value_numeric": float,
    "unit": str, "value_error": float|None, "compound": str}.
    """
    if not _CDE2_AVAILABLE:
        return []

    models = [GlassTransition, MeltingPoint]
    try:
        path = Path(html_path)
        with open(path, "rb") as f:
            doc = Document.from_file(f, fname=path.name)
        doc.models = models
    except Exception:
        return []

    props = []
    for record in doc.records:
        rec_dict = record.serialize() if hasattr(record, "serialize") else {}
        if "GlassTransition" in rec_dict:
            gt = rec_dict["GlassTransition"]
            val = gt.get("value")
            if val and isinstance(val, list):
                v = val[0] if val else None
            else:
                v = val
            unit = gt.get("units")
            if isinstance(unit, dict):
                unit = str(unit) if unit else None
            compound = ""
            if "compound" in gt and isinstance(gt["compound"], dict):
                c = gt["compound"].get("Compound", {})
                names = c.get("names", []) or c.get("labels", [])
                compound = names[0] if names else ""
            props.append({
                "property_name": "glass transition temperature",
                "property_symbol": "Tg",
                "value": str(v) if v is not None else None,
                "value_numeric": float(v) if v is not None else None,
                "unit": unit,
                "value_error": gt.get("error"),
                "compound": compound,
                "source": "cde2_rulebased",
            })
        elif "MeltingPoint" in rec_dict:
            mp = rec_dict["MeltingPoint"]
            val = mp.get("value")
            if val and isinstance(val, list):
                v = val[0] if val else None
            else:
                v = val
            unit = mp.get("units")
            if isinstance(unit, dict):
                unit = str(unit) if unit else None
            compound = ""
            if "compound" in mp and isinstance(mp["compound"], dict):
                c = mp["compound"].get("Compound", {})
                names = c.get("names", []) or c.get("labels", [])
                compound = names[0] if names else ""
            props.append({
                "property_name": "melting temperature",
                "property_symbol": "Tm",
                "value": str(v) if v is not None else None,
                "value_numeric": float(v) if v is not None else None,
                "unit": unit,
                "value_error": mp.get("error"),
                "compound": compound,
                "source": "cde2_rulebased",
            })
    return props


def merge_uncertainty_and_rulebased(
    llm_props: list[dict],
    cde2_props: list[dict],
    composition: str,
) -> list[dict]:
    """
    Merge value_error from CDE2 rule-based into LLM props when matching.
    Also add any CDE2-only properties not found by LLM.
    """
    result = list(llm_props)
    used_cde2 = set()

    for lp in result:
        pname = (lp.get("property_name") or "").lower()
        psym = (lp.get("property_symbol") or "").strip()
        # Try to find matching CDE2 prop
        for i, cp in enumerate(cde2_props):
            if i in used_cde2:
                continue
            cname = (cp.get("property_name") or "").lower()
            csym = (cp.get("property_symbol") or "").strip()
            ccomp = (cp.get("compound") or "").strip().lower()
            comp_lower = (composition or "").lower()
            if (cname in pname or pname in cname or psym == csym) and (
                not ccomp or ccomp in comp_lower or comp_lower in ccomp
            ):
                if cp.get("value_error") is not None and lp.get("value_error") is None:
                    lp["value_error"] = cp["value_error"]
                used_cde2.add(i)
                break

    # Append CDE2-only properties (ensure schema compatibility)
    for i, cp in enumerate(cde2_props):
        if i in used_cde2:
            continue
        ccomp = (cp.get("compound") or "").strip()
        if not ccomp or ccomp.lower() in (composition or "").lower():
            prop = {**cp, "source": "cde2_rulebased"}
            prop.setdefault("value_type", "exact" if prop.get("value_numeric") is not None else "missing")
            prop.setdefault("measurement_condition", None)
            prop.setdefault("additional_information", None)
            result.append(prop)
    return result
