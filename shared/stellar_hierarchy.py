from __future__ import annotations

import math
import re
from collections.abc import Mapping


ROMAN_LUMINOSITY_ORDER = ("I", "II", "III", "IV", "V")
SPECIAL_STELLAR_CLASSES = ("WD", "CARBON", "CV")
SPECIAL_TYPE_LABEL = "Special_type"
LEVEL2_CLASS_ORDER = (*ROMAN_LUMINOSITY_ORDER, SPECIAL_TYPE_LABEL)
RAW_SPECTRAL_CLASS_ORDER = ("O", "B", "A", "F", "G", "K", "M")
OB_SPECTRAL_LABEL = "OB"
SPECTRAL_CLASS_ORDER = (OB_SPECTRAL_LABEL, "A", "F", "G", "K", "M")
UNRESOLVED_LABEL = "UNRESOLVED"

GAIA_LUMINOSITY_TO_ROMAN = {
    "MAIN_SEQUENCE": "V",
    "SUBGIANT": "IV",
    "RED_GIANT_BRANCH": "III",
    "RGB_BUMP": "III",
}


def roman_luminosity_from_gaia(value) -> str:
    text = str(value or "").strip().upper()
    return GAIA_LUMINOSITY_TO_ROMAN.get(text, "")


def major_spectral_class(value) -> str:
    text = str(value or "").strip().upper()
    match = re.match(r"^\s*([OBAFGKM])", text)
    return match.group(1) if match else ""


def grouped_level2_label(value) -> str:
    text = str(value or "").strip()
    upper = text.upper()
    if upper in ROMAN_LUMINOSITY_ORDER:
        return upper
    if not upper or upper == UNRESOLVED_LABEL:
        return upper
    return SPECIAL_TYPE_LABEL


def grouped_spectral_class(value) -> str:
    text = str(value or "").strip().upper()
    if text in {"O", "B", OB_SPECTRAL_LABEL}:
        return OB_SPECTRAL_LABEL
    return text


def parse_level3_class_weights(value) -> dict[str, float]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        entries = list(value.items())
    else:
        entries = []
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(
                    "Peso de classe invalido em --hierarchy-level3-class-weights: "
                    f"{token}. Use CLASSE=VALOR."
                )
            entries.append(tuple(part.strip() for part in token.split("=", 1)))

    parsed: dict[str, float] = {}
    for raw_label, raw_multiplier in entries:
        label = grouped_spectral_class(raw_label)
        if label not in SPECTRAL_CLASS_ORDER:
            raise ValueError(
                "Classe invalida em --hierarchy-level3-class-weights: "
                f"{raw_label}"
            )
        if label in parsed:
            raise ValueError(
                "Classe repetida em --hierarchy-level3-class-weights: "
                f"{label}"
            )
        try:
            multiplier = float(raw_multiplier)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Multiplicador invalido em --hierarchy-level3-class-weights: "
                f"{raw_multiplier}"
            ) from error
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(
                "Multiplicadores de --hierarchy-level3-class-weights devem ser "
                "finitos e maiores que zero"
            )
        parsed[label] = multiplier
    return {
        label: parsed[label]
        for label in SPECTRAL_CLASS_ORDER
        if label in parsed
    }


def ordered_level2_labels(labels) -> tuple[str, ...]:
    present = {grouped_level2_label(label) for label in labels if str(label)}
    ordered = [
        label
        for label in LEVEL2_CLASS_ORDER
        if label in present
    ]
    ordered.extend(sorted(present.difference(ordered)))
    return tuple(ordered)


def ordered_spectral_labels(labels) -> tuple[str, ...]:
    present = {grouped_spectral_class(label) for label in labels if str(label)}
    ordered = [label for label in SPECTRAL_CLASS_ORDER if label in present]
    ordered.extend(sorted(present.difference(ordered)))
    return tuple(ordered)
