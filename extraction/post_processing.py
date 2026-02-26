"""
Post-processing for extracted polymer data.

- Property ontology alignment: map to controlled vocabulary (Purple Book, KnowMat)
  - Exact match from ontology JSON, or LLM-based alignment for unmatched
- Validation: value_numeric consistency, unit validation (pint)
- SI unit standardization (value_numeric_si, unit_si)
- Uncertainty extraction (value_error from value string)
- Confidence scores: heuristic per-property confidence
"""

import json
import re
from pathlib import Path

PROPERTY_ONTOLOGY_PATH = Path(__file__).parent / "property_ontology.json"
POLYMER_ABBREVIATIONS_PATH = Path(__file__).parent / "polymer_abbreviations.json"

# Unit normalization: common variants -> canonical (before pint fallback)
UNIT_ALIASES = {
    "°c": "°C", "deg c": "°C", "celsius": "°C",
    "k": "K", "kelvin": "K",
    "g/mol": "g/mol", "da": "Da", "kda": "kDa", "kg/mol": "kg/mol",
    "mpa": "MPa", "gpa": "GPa", "pa": "Pa",
    "wt%": "wt%", "mol%": "mol%", "%": "%",
}


def _load_ontology() -> dict:
    """Load property ontology mappings."""
    if not PROPERTY_ONTOLOGY_PATH.exists():
        return {"mappings": {}}
    with open(PROPERTY_ONTOLOGY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_polymer_abbreviations() -> dict[str, str]:
    """Load polymer abbreviation -> full name mappings (Distrupol, PolySource)."""
    if not POLYMER_ABBREVIATIONS_PATH.exists():
        return {}
    with open(POLYMER_ABBREVIATIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("abbreviations", {})


# Chemical element symbols: do not match as polymer abbreviations
# Br=bromide, Cl=chloride, F=fluoride, I=iodide (e.g. PEG-Br, PVC-Cl)
_ATOM_SYMBOLS = frozenset({"br", "cl", "f", "i"})


def align_composition_name(composition: str, abbreviations: dict[str, str]) -> tuple[str | None, dict[str, str]]:
    """
    Resolve polymer abbreviations in composition string.
    Returns (composition_standard, composition_abbreviations_resolved).
    - composition_standard: full name when composition is exactly an abbreviation
    - composition_abbreviations_resolved: {abbr: full_name} for abbreviations found in text
    """
    if not composition or not abbreviations:
        return (None, {})
    comp_str = (composition or "").strip()
    resolved: dict[str, str] = {}
    # Tokenize: split by common delimiters, keep alphanumeric tokens
    tokens = re.findall(r"[A-Za-z0-9]+", comp_str)
    # For each token, match longest abbreviation (avoid PE when PEO matches)
    # Skip tokens that are chemical element symbols (Br, Cl, etc.)
    abbr_sorted = sorted(abbreviations.keys(), key=len, reverse=True)
    for t in tokens:
        if t.lower() in _ATOM_SYMBOLS:
            continue  # Br=bromide, Cl=chloride, not polymer abbreviations
        t_upper = t.upper()
        for abbr in abbr_sorted:
            if abbr.upper() == t_upper or (
                t_upper.startswith(abbr.upper()) and len(t) > len(abbr)
            ):
                if abbr not in resolved:
                    resolved[abbr] = abbreviations[abbr]
                break  # longest match wins
    # Exact match: composition is just the abbreviation (possibly with spaces)
    comp_norm = re.sub(r"\s+", " ", comp_str).strip()
    abbr_upper = {k.upper(): v for k, v in abbreviations.items()}
    for abbr, full in abbr_upper.items():
        if comp_norm.upper() == abbr:
            return (full, resolved)
    return (None, resolved if resolved else {})


def align_property_name(name: str, symbol: str | None, ontology: dict) -> str:
    """
    Map property name/symbol to standard vocabulary (Purple Book, KnowMat).
    Uses only exact matches to avoid wrong mappings (e.g. extraction capacity → Young's modulus).
    """
    mappings = ontology.get("mappings", {})
    name_str = (name or "").strip()
    name_lower = name_str.lower()
    symbol_str = (symbol or "").strip()
    # Try symbol first (often more reliable) - exact match only
    if symbol_str and symbol_str in mappings:
        return mappings[symbol_str]
    # Try exact match on name (case-insensitive)
    for k, v in mappings.items():
        if k.lower() == name_lower:
            return v
    # No match - keep original (do NOT use partial match; it causes wrong mappings
    # e.g. "E" in "extraction", "D" in "midpoint" → extraction capacity→Young's modulus, pKa→dispersity)
    return name_str or ""


def _align_properties_llm_batch(
    unmatched: list[tuple[str, str | None]],
    ontology: dict,
    model: str,
) -> dict[tuple[str, str | None], str]:
    """
    Use LLM to map unmatched (name, symbol) pairs to standard vocabulary.
    Returns dict mapping (name, symbol) -> aligned_name.
    """
    if not unmatched:
        return {}
    standard_terms = sorted(set(ontology.get("mappings", {}).values()))
    items = [f"{i+1}. {n}" + (f" (symbol: {s})" if s else "") for i, (n, s) in enumerate(unmatched)]
    prompt = f"""Map these polymer/materials property names to the standard vocabulary.
If a property is equivalent to a standard term, return that term. If not, return the original name unchanged.
Do NOT map unrelated properties (e.g. extraction capacity is NOT Young's modulus, pKa is NOT dispersity).

Standard vocabulary: {', '.join(standard_terms)}

Properties to map (return aligned names in same order as JSON array):
{chr(10).join(items)}

Respond with a JSON array only, e.g. ["aligned1", "aligned2", ...]"""
    try:
        from extraction.extractors import get_extraction_llm
        llm = get_extraction_llm(model)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception:
        return {}
    # Parse JSON array from response
    match = re.search(r"\[[\s\S]*?\]", content)
    if not match:
        return {}
    try:
        aligned_list = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(aligned_list, list) or len(aligned_list) != len(unmatched):
        return {}
    out = {}
    for (name, symbol), aligned in zip(unmatched, aligned_list):
        if isinstance(aligned, str) and aligned.strip():
            out[(name, symbol)] = aligned.strip()
        else:
            out[(name, symbol)] = name
    return out


def validate_value_numeric(value: str | None, value_numeric: float | None, value_type: str) -> tuple[float | None, str | None]:
    """
    Validate value_numeric against value. Returns (corrected_value_numeric, error_msg).
    """
    if value is None or value_type == "missing":
        return (None, None) if value_numeric is None else (None, "value_numeric should be null when value is missing")
    if value_type == "qualitative":
        return (0.0, None) if value_numeric in (None, 0.0) else (value_numeric, "qualitative typically maps to 0.0")
    # Parse numeric from value
    try:
        if value_type == "exact":
            num = float(re.sub(r"[^\d.eE+-]", "", value) or "0")
            if value_numeric is not None and abs(num - value_numeric) > 0.01:
                return (num, f"value_numeric {value_numeric} inconsistent with value '{value}'")
            return (num if value_numeric is None else value_numeric, None)
        if value_type == "lower_bound":
            m = re.search(r">\s*([\d.]+)", value)
            bound = float(m.group(1)) if m else (value_numeric or 0)
            return (bound, None)
        if value_type == "upper_bound":
            m = re.search(r"<\s*([\d.]+)", value)
            bound = float(m.group(1)) if m else (value_numeric or 0)
            return (bound, None)
        if value_type == "range":
            parts = re.findall(r"[\d.]+", value)
            if len(parts) >= 2:
                mid = (float(parts[0]) + float(parts[1])) / 2
                return (mid, None)
    except (ValueError, TypeError):
        pass
    return (value_numeric, None)


# Unit -> pint-compatible string for SI conversion
_UNIT_TO_PINT = {
    "°c": "degC", "°C": "degC", "celsius": "degC", "deg c": "degC",
    "k": "K", "kelvin": "K",
    "kda": "kDa", "da": "Da", "g/mol": "g/mol", "kg/mol": "kg/mol",
    "mpa": "MPa", "gpa": "GPa", "pa": "Pa",
    "nm": "nm", "μm": "um", "å": "angstrom", "Å": "angstrom",
}


def convert_to_si(value_numeric: float | None, unit: str | None) -> tuple[float | None, str | None]:
    """
    Convert value to SI units using pint. Returns (si_value, si_unit) or (None, None).
    Temperature -> K, molar mass -> kg/mol, pressure -> Pa, length -> m.
    """
    if value_numeric is None or unit is None or not str(unit).strip():
        return (None, None)
    u = str(unit).strip().lower()
    pint_unit = _UNIT_TO_PINT.get(u) or u
    try:
        import pint
        ureg = pint.UnitRegistry()
        q = ureg.Quantity(value_numeric, pint_unit)
        # Molar mass: use kg/mol as SI (not kg)
        if u in ("g/mol", "kg/mol", "kda", "da"):
            target = ureg.kg / ureg.mol
            q_si = q.to(target)
        else:
            q_si = q.to_base_units()
        return (float(q_si.magnitude), str(q_si.units))
    except (ImportError, Exception):
        return (None, None)


def extract_uncertainty_from_value(value: str | None) -> float | None:
    """Extract uncertainty from value string (e.g. '105 ± 2' or '105(2)')."""
    if not value or not isinstance(value, str):
        return None
    # ± pattern
    m = re.search(r"±\s*([\d.]+)", value)
    if m:
        return float(m.group(1))
    # (2) pattern for uncertainty
    m = re.search(r"\((\d+)\)\s*$", value.strip())
    if m:
        return float(m.group(1))
    return None


def validate_unit(unit: str | None) -> tuple[str | None, str | None]:
    """Normalize unit and check validity. Returns (normalized_unit, error_msg)."""
    if not unit or not str(unit).strip():
        return (None, None)
    u = str(unit).strip()
    u_lower = u.lower()
    if u_lower in UNIT_ALIASES:
        return (UNIT_ALIASES[u_lower], None)
    # Try pint if available
    try:
        import pint
        ureg = pint.UnitRegistry()
        parsed = ureg(u)
        return (str(parsed.units), None)
    except ImportError:
        return (u, None)
    except Exception:
        return (u, f"unit '{u}' may be invalid")


def compute_confidence(prop: dict) -> float:
    """
    Heuristic confidence score 0-1 for a property.
    - Has value + unit + value_numeric: higher
    - Has value_type exact: higher
    - Has value_error: higher (measurement uncertainty reported)
    - Missing value or qualitative: lower
    """
    score = 0.5  # base
    vtype = prop.get("value_type", "exact")
    if prop.get("value") and vtype != "missing":
        score += 0.2
    if prop.get("unit"):
        score += 0.1
    if prop.get("value_numeric") is not None and vtype == "exact":
        score += 0.2
    if prop.get("measurement_condition"):
        score += 0.05
    if prop.get("property_symbol"):
        score += 0.05
    if prop.get("value_error") is not None:
        score += 0.05
    return min(1.0, score)


def post_process_compositions(
    data: dict,
    align_ontology: bool = True,
    align_compositions: bool = True,
    validate: bool = True,
    add_confidence: bool = True,
    add_si: bool = True,
    add_uncertainty: bool = True,
    use_llm_alignment: bool = False,
    alignment_model: str | None = None,
) -> dict:
    """Apply all post-processing to extracted compositions."""
    ontology = _load_ontology() if align_ontology else {}
    polymer_abbr = _load_polymer_abbreviations() if align_compositions else {}
    # Composition alignment: resolve polymer abbreviations (Distrupol, PolySource)
    for comp in data.get("compositions", []):
        if align_compositions and polymer_abbr:
            comp_str = comp.get("composition", "")
            standard, resolved = align_composition_name(comp_str, polymer_abbr)
            if standard:
                comp["composition_standard"] = standard
            if resolved:
                comp["composition_abbreviations_resolved"] = resolved
    # First pass: exact match, collect unmatched for LLM
    unmatched: list[tuple[str, str | None]] = []
    unmatched_props: list[dict] = []
    for comp in data.get("compositions", []):
        for prop in comp.get("properties_of_composition", []):
            if not align_ontology or not ontology.get("mappings"):
                continue
            original = prop.get("property_name", "")
            symbol = prop.get("property_symbol")
            aligned = align_property_name(original, symbol, ontology)
            if aligned != original:
                prop["property_name_original"] = original
                prop["property_name"] = aligned
            elif use_llm_alignment and alignment_model and original:
                key = (original, symbol)
                if key not in [(u[0], u[1]) for u in unmatched]:
                    unmatched.append(key)
                unmatched_props.append(prop)
    # LLM alignment for unmatched
    if use_llm_alignment and alignment_model and unmatched:
        llm_mappings = _align_properties_llm_batch(unmatched, ontology, alignment_model)
        for prop in unmatched_props:
            original = prop.get("property_name", "")
            symbol = prop.get("property_symbol")
            key = (original, symbol)
            aligned = llm_mappings.get(key, original)
            if aligned != original:
                prop["property_name_original"] = original
                prop["property_name"] = aligned
    # Second pass: validation, SI, uncertainty, confidence
    for comp in data.get("compositions", []):
        for prop in comp.get("properties_of_composition", []):
            # Validation
            if validate:
                vn, err = validate_value_numeric(
                    prop.get("value"),
                    prop.get("value_numeric"),
                    prop.get("value_type", "exact"),
                )
                if vn is not None:
                    prop["value_numeric"] = vn
                if err:
                    prop["validation_warning"] = err
                u, uerr = validate_unit(prop.get("unit"))
                if u is not None:
                    prop["unit"] = u
                if uerr:
                    prop["unit_warning"] = uerr
            # SI conversion
            if add_si and prop.get("value_numeric") is not None and prop.get("unit"):
                si_val, si_unit = convert_to_si(prop["value_numeric"], prop["unit"])
                if si_val is not None:
                    prop["value_numeric_si"] = round(si_val, 6)
                    prop["unit_si"] = si_unit
            # Uncertainty from value string (if not already set)
            if add_uncertainty and prop.get("value_error") is None:
                err = extract_uncertainty_from_value(prop.get("value"))
                if err is not None:
                    prop["value_error"] = err
            # Confidence
            if add_confidence:
                prop["confidence"] = round(compute_confidence(prop), 2)
    return data
