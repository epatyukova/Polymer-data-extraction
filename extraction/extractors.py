"""
TrustCall extractor definitions (KnowMat-style schema).

Uses Pydantic models for schema enforcement via TrustCall.
Schema: compositions (polymer names), characterisation, properties_of_composition
with full property encoding: value_numeric, measurement_condition, additional_information, property_symbol.

Supports OpenAI, Anthropic, and Ollama (ollama:deepseek-r1:8b) models.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from trustcall import create_extractor


def get_extraction_llm(model: str = "gpt-5.2"):
    """
    Create LLM for extraction.
    - OpenAI: gpt-4o, gpt-4.1, gpt-5.2
    - Anthropic: claude-3-5-sonnet, claude-3-opus
    - Ollama (local): ollama:deepseek-r1:8b, ollama:llama3.1
    """
    model_lower = model.lower()
    if model_lower.startswith("ollama:"):
        from langchain_ollama import ChatOllama
        ollama_model = model[7:].strip()  # strip "ollama:"
        return ChatOllama(
            model=ollama_model,
            temperature=0.1,
            num_ctx=32768,
        )
    if model_lower.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=0.1,
            max_tokens=16384,
        )
    # OpenAI
    from langchain_openai import ChatOpenAI
    if any(x in model for x in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]):
        return ChatOpenAI(
            model=model,
            request_timeout=1200,
            max_retries=3,
        )
    return ChatOpenAI(
        model=model,
        temperature=0.1,
        request_timeout=1200,
        max_retries=3,
    )


# -----------------------------------------------------------------------------
# Pydantic schemas (KnowMat-style)
# -----------------------------------------------------------------------------


class Property(BaseModel):
    """Material property with ML-ready encoding."""

    property_name: str = Field(
        description=(
            "Full descriptive name (e.g. 'glass transition temperature', "
            "'melting temperature', 'number-average molecular weight')."
        )
    )
    property_symbol: Optional[str] = Field(
        default=None,
        description=(
            "Standard symbol as used in paper (e.g. 'Tg', 'Tm', 'Mw', 'Mn'). "
            "Use null if no symbol provided."
        )
    )
    value: Optional[str] = Field(
        default=None,
        description=(
            "Original value from paper. Can be: "
            "(1) numeric string '683.0', (2) inequality '>50' or '<2000', "
            "(3) range '12-30', (4) qualitative 'brittle', (5) null for missing."
        )
    )
    value_numeric: Optional[float] = Field(
        default=None,
        description=(
            "ML-ready numeric. Exact: 683.0. Inequality: '>50' → 50.0. "
            "Range: '12-30' → 21.0. Qualitative: 'brittle' → 0.0. Missing: null."
        )
    )
    value_type: str = Field(
        description=(
            "One of: exact, lower_bound, upper_bound, range, qualitative, missing."
        )
    )
    unit: Optional[str] = Field(
        default=None,
        description="Measurement unit (e.g. 'K', '°C', 'MPa', 'g/mol', 'g/mol')."
    )
    measurement_condition: Optional[str] = Field(
        default=None,
        description=(
            "Conditions under which measured (temperature, sample geometry, "
            "heating rate, technique). Use null if not provided."
        )
    )
    additional_information: Optional[str] = Field(
        default=None,
        description=(
            "Figure/table references, citations, uncertainty notes. Use null if none."
        )
    )
    value_error: Optional[float] = Field(
        default=None,
        description=(
            "Measurement uncertainty (e.g. ±2 in '105 ± 2 °C'). Use null if not reported."
        )
    )


class CompositionProperties(BaseModel):
    """One polymer/composition with processing, characterisation, and properties."""

    composition: str = Field(
        description=(
            "Polymer name: IUPAC or common (e.g. 'poly(styrene)', 'PMMA', "
            "'polystyrene-b-poly(methyl methacrylate)')."
        )
    )
    processing_conditions: str = Field(
        description=(
            "Synthesis and processing: temperature, time, atmosphere, "
            "method (RAFT, ATRP, etc.). Use 'not provided' if absent."
        )
    )
    characterisation: Dict[str, str] = Field(
        description="Characterisation techniques and findings, keyed by technique name."
    )
    properties_of_composition: List[Property] = Field(
        description="List of extracted properties for this composition."
    )


class CompositionList(BaseModel):
    """List of compositions (polymers) with full extraction data."""

    compositions: List[CompositionProperties] = Field(
        description="List of extracted polymer compositions."
    )


# -----------------------------------------------------------------------------
# Extraction: TrustCall (for tool-supporting models) or Ollama JSON (for others)
# -----------------------------------------------------------------------------

# Ollama models that use JSON fallback (no tool calling or tool support unreliable)
_OLLAMA_JSON_FALLBACK = {"deepseek-r1", "deepseek-r1:8b", "deepseek-r1:14b", "llama3.1", "llama3.1:8b"}


def _is_ollama_json_fallback(model: str) -> bool:
    """True if this Ollama model should use JSON fallback (no tool support)."""
    m = model.lower().replace("ollama:", "").strip()
    return any(m.startswith(x) or x in m for x in _OLLAMA_JSON_FALLBACK)


def _extract_with_ollama_json(prompt: str, model: str) -> dict:
    """
    Extract using Ollama with JSON schema (no tool calling).
    Used for models like deepseek-r1:8b that don't support tools.
    """
    import json
    import re

    ollama_model = model.replace("ollama:", "").strip()
    try:
        from ollama import chat
    except ImportError:
        return {"compositions": []}

    schema = CompositionList.model_json_schema()
    try:
        response = chat(
            model=ollama_model,
            messages=[{"role": "user", "content": prompt}],
            format=schema,
        )
    except Exception:
        try:
            response = chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt + "\n\nReply with valid JSON only, no other text."}],
            )
        except Exception:
            return {"compositions": []}

    content = ""
    if isinstance(response, dict):
        content = response.get("message", {}).get("content", "")
    else:
        content = getattr(getattr(response, "message", None), "content", "") or ""

    if not content.strip():
        return {"compositions": []}

    # Parse JSON - may be wrapped in markdown code block
    content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE)
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            data = json.loads(match.group(0))
            comps = data.get("compositions", [])
            return {"compositions": comps}
        except json.JSONDecodeError:
            pass
    return {"compositions": []}


def _create_extractor(model: str = "gpt-5.2"):
    """Create extraction extractor with fresh LLM."""
    llm = get_extraction_llm(model)
    return create_extractor(
        llm,
        tools=[CompositionList],
        tool_choice="CompositionList",
        enable_inserts=True,
    )


def _extract_with_direct_json(prompt: str, model: str) -> dict:
    """
    Fallback: use LLM directly (no tools), ask for JSON, parse response.
    Used when TrustCall fails (e.g. JsonPointerException from tool-call format mismatch).
    """
    import json
    import re

    suffix = "\n\nRespond with valid JSON only. Use the format: {\"compositions\": [...]}. No other text."
    try:
        llm = get_extraction_llm(model)
        response = llm.invoke(prompt + suffix)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception:
        return {"compositions": []}

    if not content or not content.strip():
        return {"compositions": []}

    content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE)
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            data = json.loads(match.group(0))
            comps = data.get("compositions", [])
            return {"compositions": comps}
        except json.JSONDecodeError:
            pass
    return {"compositions": []}


def extract_with_trustcall(prompt: str, model: str = "gpt-5.2") -> dict:
    """
    Run extraction. For ollama: models without tool support (e.g. deepseek-r1),
    uses JSON schema fallback. Otherwise uses TrustCall.
    """
    model_lower = model.lower()
    if model_lower.startswith("ollama:") and _is_ollama_json_fallback(model):
        return _extract_with_ollama_json(prompt, model)

    try:
        extractor = _create_extractor(model)
        result = extractor.invoke({
            "messages": [{"role": "user", "content": prompt}]
        })
        responses = result.get("responses", [None])
        response = responses[0] if responses else None
        if response is None:
            return {"compositions": []}
        if isinstance(response, CompositionList):
            return response.model_dump()
        return dict(response)
    except Exception as e:
        err = str(e).lower()
        # TrustCall/JSON patch error (e.g. JsonPointerException: "arguments" not found)
        try:
            from jsonpointer import JsonPointerException
            if isinstance(e, JsonPointerException):
                return _extract_with_direct_json(prompt, model)
        except ImportError:
            pass
        if "jsonpointer" in err or "arguments" in err or "member" in err:
            return _extract_with_direct_json(prompt, model)
        if "does not support tools" in err or "tool" in err:
            if model_lower.startswith("ollama:"):
                return _extract_with_ollama_json(prompt, model)
        raise
