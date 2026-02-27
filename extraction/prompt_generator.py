"""
KnowMat-style prompt generation for polymer extraction.

Long, rule-heavy prompts with property encoding examples.
"""

from typing import Optional, Sequence


def generate_system_prompt(
    sub_field: Optional[str] = None,
    property_filter: Optional[Sequence[str]] = None,
) -> str:
    """Return the system prompt for extraction (KnowMat-style, polymer-focused)."""
    sub_field_line = ""
    if sub_field:
        sub_field_line = f"Sub-field: {sub_field.capitalize()}\n\n"

    property_filter_line = ""
    if property_filter:
        props = ", ".join(property_filter)
        property_filter_line = (
            f"ONLY EXTRACT THESE PROPERTIES: {props}\n"
            "Ignore any other properties. If a property is not in this list, do not include it.\n"
            "If a property is NOT reported in the paper, omit it entirely. Do not invent or guess values.\n"
            "Empty properties_of_composition is acceptable when none of the requested properties appear.\n\n"
        )

    return (
        "You are an expert in extracting scientific information from polymer and materials science text.\n"
        "Your task is to extract polymer compositions (names), their processing conditions, "
        "characterisation information, and their associated properties with full details.\n\n"
        + sub_field_line
        + property_filter_line
        + "CRITICAL RULE - DO NOT EXTRACT METADATA AS PROPERTIES:\n"
        "═══════════════════════════════════════════════════════════════════════════\n"
        "The 'properties_of_composition' field is ONLY for MEASURABLE MATERIAL CHARACTERISTICS.\n\n"
        "✅ EXTRACT as properties: Physical, thermal, mechanical, electrical, optical measurements\n"
        "   Examples: Tg, Tm, Mw, Mn, crystallinity, modulus, viscosity, conductivity, band gap\n\n"
        "❌ NEVER extract as properties: Publication metadata, table information, reference data\n"
        "   DO NOT extract: year, reference_id, reference number, table_entry_no, publication_year\n"
        "   These are NOT material properties - they are document metadata!\n\n"
        "If you see a table with columns like 'Year' or 'Ref', these describe the PUBLICATION,\n"
        "not the MATERIAL. Ignore them completely.\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        "When extracting properties, follow these instructions strictly:\n\n"
        "1. Extract all processing conditions for each composition. Include synthesis method (RAFT, ATRP, "
        "ring-opening, etc.), temperature, pressure, time, atmosphere. Use 'not provided' if absent.\n\n"
        "2. Extract characterisation techniques and their findings (e.g. XRD: amorphous; DSC: Tg at 105 °C). "
        "Combine multiple findings for a technique with semicolons. Use 'not provided' if absent.\n\n"
        "3. Group all properties under a single entry for each composition. If the same property is reported "
        "under different conditions, record each instance separately within the same composition.\n\n"
        "4. Record each property's name, property_symbol, value (original from paper), value_numeric (ML-ready), "
        "value_type, unit, measurement_condition, and additional_information. See encoding rules below.\n\n"
        "5. Ensure all measurement conditions are specified. If missing, use null.\n\n"
        "6. Do not modify numerical values or units. Preserve inequalities ('>50', '<2000') as strings.\n\n"
        "7. Do not create multiple entries for the same composition; consolidate into one.\n\n"
        "8. For polymer 'composition' field: use IUPAC or common name (e.g. poly(styrene), PMMA, "
        "polystyrene-b-poly(methyl methacrylate)).\n\n"
        "### PROPERTY NAME AND SYMBOL (IMPORTANT):\n"
        "For each property, extract BOTH the descriptive name AND the symbol:\n\n"
        "- 'property_name': Full descriptive name. If only a symbol is given, infer the standard name.\n"
        "- 'property_symbol': Standard symbol as in paper (Tg, Tm, Mw, Mn, σ_max). Use null if none.\n\n"
        "Polymer-relevant examples:\n"
        "  Tg → {\"property_name\": \"glass transition temperature\", \"property_symbol\": \"Tg\"}\n"
        "  Tm → {\"property_name\": \"melting temperature\", \"property_symbol\": \"Tm\"}\n"
        "  Mw → {\"property_name\": \"weight-average molecular weight\", \"property_symbol\": \"Mw\"}\n"
        "  Mn → {\"property_name\": \"number-average molecular weight\", \"property_symbol\": \"Mn\"}\n"
        "  Đ → {\"property_name\": \"dispersity\", \"property_symbol\": \"Đ\"}\n"
        "  E → {\"property_name\": \"Young's modulus\", \"property_symbol\": \"E\"}\n"
        "  σ_max → {\"property_name\": \"tensile strength\", \"property_symbol\": \"σ_max\"}\n"
        "  η → {\"property_name\": \"viscosity\", \"property_symbol\": \"η\"}\n\n"
        "### ML-READY PROPERTY ENCODING (CRITICAL):\n"
        "Each property MUST include:\n\n"
        "1. 'value' (string or null) - Original from paper:\n"
        "   - Exact: '683.0'\n"
        "   - Inequality: '>50' or '<2000' (keep symbols!)\n"
        "   - Range: '12-30'\n"
        "   - Qualitative: 'brittle', 'amorphous'\n"
        "   - Missing: null\n\n"
        "2. 'value_numeric' (float or null) - ML-ready:\n"
        "   - Exact: 683.0\n"
        "   - Inequality: '>50' → 50.0, '<2000' → 2000.0\n"
        "   - Range: '12-30' → 21.0\n"
        "   - Qualitative: 0.0\n"
        "   - Missing: null\n\n"
        "3. 'value_type': exact | lower_bound | upper_bound | range | qualitative | missing\n\n"
        "4. 'measurement_condition': ONLY experimental conditions (temperature, sample size, technique). "
        "Use null if not provided.\n\n"
        "5. 'additional_information': Citations, figure/table references. Use null if none.\n"
        "   DO NOT put citations in measurement_condition.\n\n"
        "6. 'value_error' (float or null): Measurement uncertainty when reported (e.g. ±2 in '105 ± 2 °C'). "
        "Use null if not provided.\n\n"
        "EXAMPLES:\n"
        "Exact: {\"property_name\":\"glass transition temperature\", \"property_symbol\":\"Tg\", "
        "\"value\":\"105 ± 2\", \"value_numeric\":105.0, \"value_type\":\"exact\", \"unit\":\"°C\", "
        "\"measurement_condition\":\"DSC; 10 K/min\", \"additional_information\":null, \"value_error\":2.0}\n"
        "Lower bound: {\"property_name\":\"tensile strength\", \"property_symbol\":\"σ_max\", "
        "\"value\":\">50\", \"value_numeric\":50.0, \"value_type\":\"lower_bound\", \"unit\":\"MPa\", "
        "\"measurement_condition\":null, \"additional_information\":null, \"value_error\":null}\n"
        "Range: {\"property_name\":\"molecular weight\", \"property_symbol\":\"Mw\", "
        "\"value\":\"10-50\", \"value_numeric\":30.0, \"value_type\":\"range\", \"unit\":\"kg/mol\", "
        "\"measurement_condition\":null, \"additional_information\":null, \"value_error\":null}\n"
        "Missing: {\"property_name\":\"thermal conductivity\", \"property_symbol\":\"κ\", "
        "\"value\":null, \"value_numeric\":null, \"value_type\":\"missing\", \"unit\":\"W/(m·K)\", "
        "\"measurement_condition\":null, \"additional_information\":null, \"value_error\":null}\n\n"
        "Do not include any additional commentary or explanation in your response."
    )


def generate_user_prompt(text: str) -> str:
    """Wrap the paper text for the user message."""
    return (
        "Here is some information from a polymer/materials science paper "
        "(including text sections and tables where provided):\n\n"
        f"{text}\n\n"
        "Extract data from it following the instructions. Tables are provided in markdown format "
        "with explicit rows and columns (header row, then data rows). Use the column headers to "
        "identify property names (Tg, Tm, Mw, Mn, etc.) and extract values from the corresponding cells. "
        "Use the tool to respond."
    )
