from __future__ import annotations

import re

import pandas as pd


def is_missing_label(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def normalize_spectral_label(raw_label) -> dict:
    raw = "UNKNOWN" if is_missing_label(raw_label) else str(raw_label).strip()
    upper = raw.upper().strip()
    cleaned = re.sub(r"\([^)]*\)", "", upper)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned in {"", "SEM SUBCLASSE", "SEM_SUBCLASSE", "UNKNOWN", "NAN"}:
        parent = "UNKNOWN"
        leaf = "UNKNOWN"
    elif "WD" in cleaned:
        parent = "WD"
        leaf = "WD"
    elif "CARBON" in cleaned:
        parent = "CARBON"
        leaf = "CARBON"
    elif cleaned == "CV" or cleaned.startswith("CV "):
        parent = "CV"
        leaf = "CV"
    elif cleaned == "OB":
        parent = "OB"
        leaf = "OB"
    else:
        compact = cleaned.replace(":", "")
        compact = re.sub(r"^SD\s*", "", compact)
        compact = re.sub(r"^D\s*", "", compact)
        match = re.search(r"([OBAFGKMLT])\s*([0-9](?:\.[0-9])?)?", compact)
        if match:
            parent = match.group(1)
            subtype = match.group(2)
            leaf = f"{parent}{subtype}" if subtype else parent
        else:
            parent = "UNKNOWN"
            leaf = "UNKNOWN"

    return {
        "spectral_subclass_raw": raw,
        "spectral_parent_label": parent,
        "spectral_leaf_label": leaf,
    }
