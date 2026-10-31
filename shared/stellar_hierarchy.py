from __future__ import annotations

import re


ROMAN_LUMINOSITY_ORDER = ("I", "II", "III", "IV", "V")
SPECIAL_STELLAR_CLASSES = ("WD", "CARBON", "CV")
SPECTRAL_CLASS_ORDER = ("O", "B", "A", "F", "G", "K", "M")
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


def ordered_level2_labels(labels) -> tuple[str, ...]:
    present = {str(label) for label in labels if str(label)}
    ordered = [
        label
        for label in (*ROMAN_LUMINOSITY_ORDER, *SPECIAL_STELLAR_CLASSES)
        if label in present
    ]
    ordered.extend(sorted(present.difference(ordered)))
    return tuple(ordered)


def ordered_spectral_labels(labels) -> tuple[str, ...]:
    present = {str(label) for label in labels if str(label)}
    ordered = [label for label in SPECTRAL_CLASS_ORDER if label in present]
    ordered.extend(sorted(present.difference(ordered)))
    return tuple(ordered)
