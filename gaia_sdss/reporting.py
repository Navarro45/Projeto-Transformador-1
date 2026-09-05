from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaia_sdss.config import GAIA_HIGH_PM_MAS_YR
from gaia_sdss.matching import compare_classes, derive_gaia_dsc_class, derive_spectral_class_from_teff, proper_motion_total
from gaia_sdss.stellar import derive_stellar_evolution_class


MATCH_COLUMNS = [
    "source_id",
    "designation",
    "ra",
    "dec",
    "ref_epoch",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "ruwe",
    "visibility_periods_used",
    "astrometric_excess_noise",
    "classprob_dsc_combmod_star",
    "classprob_dsc_combmod_quasar",
    "classprob_dsc_combmod_galaxy",
]


def enrich_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    derived_rows = []
    for _, row in df.iterrows():
        payload = row.to_dict()
        teff = payload.get("gaia_teff_gspphot")
        if teff is None and "teff_gspphot" in payload:
            teff = payload.get("teff_gspphot")
        payload.update(derive_spectral_class_from_teff(teff))
        payload.update(derive_gaia_dsc_class({
            "classprob_dsc_combmod_star": payload.get("gaia_classprob_dsc_combmod_star"),
            "classprob_dsc_combmod_quasar": payload.get("gaia_classprob_dsc_combmod_quasar"),
            "classprob_dsc_combmod_galaxy": payload.get("gaia_classprob_dsc_combmod_galaxy"),
        }))
        payload.update(derive_stellar_evolution_class(payload))
        payload.update(compare_classes(payload))
        pm_total = proper_motion_total(payload.get("gaia_pmra"), payload.get("gaia_pmdec"))
        payload["gaia_pm_total_mas_yr"] = pm_total
        payload["gaia_high_pm_flag"] = bool(pm_total is not None and pm_total >= GAIA_HIGH_PM_MAS_YR)
        derived_rows.append(payload)

    return pd.DataFrame(derived_rows)


def build_consolidated_dataset(local_df: pd.DataFrame, gaia_df: pd.DataFrame) -> pd.DataFrame:
    if gaia_df.empty:
        gaia_df = pd.DataFrame(columns=["specObjID", "match_status"])

    local = local_df.copy()
    local["specObjID"] = local["specObjID"].astype(str)
    gaia = gaia_df.copy()
    gaia["specObjID"] = gaia["specObjID"].astype(str)

    rename = {column: f"gaia_{column}" for column in MATCH_COLUMNS if column in gaia.columns}
    gaia = gaia.rename(columns=rename)

    merged = local.merge(gaia, on="specObjID", how="left", suffixes=("", "_gaia_match"))
    if "objID_gaia_match" in merged.columns:
        merged = merged.drop(columns=["objID_gaia_match"])

    return enrich_derived_columns(merged)


def coverage_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for column in columns:
        if column not in df.columns:
            rows.append({"parameter": column, "available": 0, "missing": total, "coverage_percent": 0.0})
            continue
        available = int(df[column].notna().sum())
        rows.append(
            {
                "parameter": column,
                "available": available,
                "missing": int(total - available),
                "coverage_percent": float((available / total * 100.0) if total else 0.0),
            }
        )
    return pd.DataFrame(rows)


def numeric_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "parameter": column,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p05": float(values.quantile(0.05)),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _save_bar(series: pd.Series, output_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    if series.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax = series.sort_values(ascending=False).head(30).plot(kind="bar", figsize=(12, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_hist(values: pd.Series, output_path: Path, title: str, xlabel: str) -> None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_heatmap(table: pd.DataFrame, output_path: Path, title: str) -> None:
    if table.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(7, len(table.columns) * 0.8), max(5, len(table.index) * 0.45)))
    plt.imshow(table.values, aspect="auto", cmap="viridis")
    plt.title(title)
    plt.xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
    plt.yticks(range(len(table.index)), table.index)
    plt.colorbar(label="Quantidade")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _markdown_table(frame: pd.DataFrame | pd.Series, max_rows: int | None = None) -> str:
    if isinstance(frame, pd.Series):
        frame = frame.reset_index()
        frame.columns = ["value", "count"]
    if frame.empty:
        return "Sem dados."
    if max_rows is not None:
        frame = frame.head(max_rows)

    text = frame.copy()
    text = text.fillna("")
    columns = [str(column) for column in text.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in text.iterrows():
        values = [str(row[column]).replace("|", "/") for column in text.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _nonempty_string_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=str)
    series = df[column].fillna("").astype(str).str.strip()
    return series[series != ""]


def _nonempty_string_pair(df: pd.DataFrame, left: str, right: str) -> tuple[pd.Series, pd.Series]:
    if left not in df.columns or right not in df.columns:
        return pd.Series(dtype=str), pd.Series(dtype=str)
    left_series = df[left].fillna("").astype(str).str.strip()
    right_series = df[right].fillna("").astype(str).str.strip()
    mask = (left_series != "") & (right_series != "")
    return left_series[mask], right_series[mask]


def generate_reports(df: pd.DataFrame, output_dir: str | Path, sources: dict) -> dict:
    df = enrich_derived_columns(df)
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    total = len(df)
    match_status_counts = df.get("match_status", pd.Series(dtype=str)).fillna("NOT_QUERIED").value_counts()
    matched_mask = df.get("match_status", pd.Series(dtype=str)).isin(["MATCH_UNIQUE", "MATCH_BEST", "MATCH_AMBIGUOUS"])

    gaia_params = [f"gaia_{column}" for column in MATCH_COLUMNS if f"gaia_{column}" in df.columns]
    coverage = coverage_table(df, gaia_params)
    coverage.to_csv(reports_dir / "gaia_parameter_coverage.csv", index=False)

    stats_columns = [
        "best_distance_arcsec",
        "gaia_teff_gspphot",
        "gaia_logg_gspphot",
        "gaia_mh_gspphot",
        "gaia_parallax",
        "absolute_g_mag",
        "parallax_over_error",
        "parallax_fractional_error",
        "gaia_pm_total_mas_yr",
        "gaia_ruwe",
    ]
    stats = numeric_stats(df, stats_columns)
    stats.to_csv(reports_dir / "numeric_stats.csv", index=False)

    subclass_counts = df.get("pipeline_subclass", pd.Series(dtype=str)).fillna("SEM_SUBCLASSE").value_counts()
    spectral_subclass_counts = df.get("spectral_subclass_raw", pd.Series(dtype=str)).fillna("SEM_SUBCLASSE").value_counts()
    spectral_major_counts = df.get("spectral_class_major", pd.Series(dtype=str)).fillna("UNKNOWN").value_counts()
    stellar_evolution_counts = df.get("stellar_evolution_class", pd.Series(dtype=str)).fillna("UNKNOWN").value_counts()
    parallax_quality_counts = df.get("parallax_quality_flag", pd.Series(dtype=str)).fillna("MISSING").value_counts()
    gaia_derived_counts = _nonempty_string_series(df, "gaia_derived_class").value_counts()
    pipeline_major, gaia_derived = _nonempty_string_pair(df, "pipeline_subclass_major", "gaia_derived_class")
    sdss_class, gaia_dsc = _nonempty_string_pair(df, "spectralClass", "gaia_dsc_class")
    spectral_major, stellar_evolution = _nonempty_string_pair(df, "spectral_class_major", "stellar_evolution_class")
    pipeline_subclass, subclass_evolution = _nonempty_string_pair(df, "pipeline_subclass", "stellar_evolution_class")
    comparison_matrix = pd.crosstab(
        pipeline_major,
        gaia_derived,
    )
    dsc_matrix = pd.crosstab(
        sdss_class,
        gaia_dsc,
    )
    spectral_evolution_matrix = pd.crosstab(
        spectral_major,
        stellar_evolution,
    )
    subclass_evolution_matrix = pd.crosstab(
        pipeline_subclass,
        subclass_evolution,
    )

    match_status_counts.to_csv(reports_dir / "match_status_counts.csv", header=["count"])
    subclass_counts.to_csv(reports_dir / "pipeline_subclass_counts.csv", header=["count"])
    spectral_subclass_counts.to_csv(reports_dir / "spectral_subclass_raw_counts.csv", header=["count"])
    spectral_major_counts.to_csv(reports_dir / "spectral_class_major_counts.csv", header=["count"])
    stellar_evolution_counts.to_csv(reports_dir / "stellar_evolution_class_counts.csv", header=["count"])
    parallax_quality_counts.to_csv(reports_dir / "parallax_quality_counts.csv", header=["count"])
    gaia_derived_counts.to_csv(reports_dir / "gaia_derived_class_counts.csv", header=["count"])
    comparison_matrix.to_csv(reports_dir / "spectral_comparison_matrix.csv")
    dsc_matrix.to_csv(reports_dir / "dsc_comparison_matrix.csv")
    spectral_evolution_matrix.to_csv(reports_dir / "spectral_evolution_matrix.csv")
    subclass_evolution_matrix.to_csv(reports_dir / "subclass_evolution_matrix.csv")

    _save_bar(match_status_counts, plots_dir / "match_status_counts.png", "Status do cross-match Gaia", "Status", "Quantidade")
    if "best_distance_arcsec" in df.columns:
        _save_hist(df.loc[matched_mask, "best_distance_arcsec"], plots_dir / "match_distance_histogram.png", "Distancias do cross-match", "Distancia angular (arcsec)")
    _save_bar(subclass_counts, plots_dir / "pipeline_subclass_counts.png", "Subclasses do pipeline local", "Subclasse", "Quantidade")
    _save_bar(spectral_subclass_counts, plots_dir / "spectral_subclass_raw_counts.png", "Subclasses espectrais SDSS", "Subclasse", "Quantidade")
    _save_bar(spectral_major_counts, plots_dir / "spectral_class_major_counts.png", "Classe espectral ampla", "Classe", "Quantidade")
    _save_bar(stellar_evolution_counts, plots_dir / "stellar_evolution_class_counts.png", "Classe evolutiva/fisica derivada", "Classe", "Quantidade")
    _save_bar(parallax_quality_counts, plots_dir / "parallax_quality_counts.png", "Qualidade da paralaxe Gaia", "Status", "Quantidade")
    _save_bar(gaia_derived_counts, plots_dir / "gaia_derived_class_counts.png", "Classes derivadas do Gaia por Teff", "Classe derivada", "Quantidade")
    _save_heatmap(comparison_matrix, plots_dir / "spectral_comparison_matrix.png", "Pipeline subclass x Gaia Teff")
    _save_heatmap(dsc_matrix, plots_dir / "dsc_comparison_matrix.png", "Classe SDSS x Gaia DSC")
    _save_heatmap(spectral_evolution_matrix, plots_dir / "spectral_evolution_matrix.png", "Classe espectral ampla x classe evolutiva")
    _save_heatmap(subclass_evolution_matrix, plots_dir / "subclass_evolution_matrix.png", "Subclasse SDSS x classe evolutiva")

    if {"gaia_bp_rp", "gaia_phot_g_mean_mag"}.issubset(df.columns):
        phot = df[["gaia_bp_rp", "gaia_phot_g_mean_mag"]].apply(pd.to_numeric, errors="coerce").dropna()
        if not phot.empty:
            plt.figure(figsize=(7, 6))
            plt.scatter(phot["gaia_bp_rp"], phot["gaia_phot_g_mean_mag"], s=8, alpha=0.5)
            plt.gca().invert_yaxis()
            plt.title("Diagrama fotometrico Gaia")
            plt.xlabel("BP - RP (mag)")
            plt.ylabel("G (mag)")
            plt.tight_layout()
            plt.savefig(plots_dir / "gaia_color_magnitude.png", dpi=160)
            plt.close()

    if {"gaia_bp_rp", "absolute_g_mag"}.issubset(df.columns):
        hr = df[["gaia_bp_rp", "absolute_g_mag", "stellar_evolution_class"]].copy()
        hr["gaia_bp_rp"] = pd.to_numeric(hr["gaia_bp_rp"], errors="coerce")
        hr["absolute_g_mag"] = pd.to_numeric(hr["absolute_g_mag"], errors="coerce")
        hr = hr.dropna(subset=["gaia_bp_rp", "absolute_g_mag"])
        if not hr.empty:
            plt.figure(figsize=(8, 7))
            for label, group in hr.groupby("stellar_evolution_class"):
                plt.scatter(group["gaia_bp_rp"], group["absolute_g_mag"], s=10, alpha=0.6, label=label)
            plt.gca().invert_yaxis()
            plt.title("Diagrama HR Gaia derivado")
            plt.xlabel("BP - RP (mag)")
            plt.ylabel("Magnitude absoluta G (mag)")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plots_dir / "gaia_hr_diagram.png", dpi=160)
            plt.close()

    if {"gaia_teff_gspphot", "stellar_evolution_class"}.issubset(df.columns):
        temp = df[["gaia_teff_gspphot", "stellar_evolution_class"]].copy()
        temp["gaia_teff_gspphot"] = pd.to_numeric(temp["gaia_teff_gspphot"], errors="coerce")
        temp = temp.dropna(subset=["gaia_teff_gspphot"])
        groups = [
            group["gaia_teff_gspphot"].to_numpy()
            for _, group in temp.groupby("stellar_evolution_class")
            if not group.empty
        ]
        labels = [
            label
            for label, group in temp.groupby("stellar_evolution_class")
            if not group.empty
        ]
        if groups:
            plt.figure(figsize=(10, 5))
            try:
                plt.boxplot(groups, tick_labels=labels, showfliers=False)
            except TypeError:
                plt.boxplot(groups, labels=labels, showfliers=False)
            plt.title("Temperatura Gaia por classe evolutiva")
            plt.xlabel("Classe evolutiva/fisica")
            plt.ylabel("Teff Gaia (K)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(plots_dir / "teff_by_stellar_evolution_class.png", dpi=160)
            plt.close()

    summary = {
        "sources": sources,
        "total_local_objects": int(total),
        "objects_with_radec": int(df.get("has_radec", pd.Series(dtype=bool)).fillna(False).sum()),
        "total_with_match": int(matched_mask.sum()),
        "total_no_match": int((df.get("match_status", pd.Series(dtype=str)) == "NO_MATCH").sum()),
        "total_ambiguous": int((df.get("match_status", pd.Series(dtype=str)) == "MATCH_AMBIGUOUS").sum()),
        "total_errors": int((df.get("match_status", pd.Series(dtype=str)) == "ERROR").sum()),
        "match_status_counts": match_status_counts.to_dict(),
        "stellar_evolution_class_counts": stellar_evolution_counts.to_dict(),
        "parallax_quality_counts": parallax_quality_counts.to_dict(),
        "coverage_csv": str(reports_dir / "gaia_parameter_coverage.csv"),
        "numeric_stats_csv": str(reports_dir / "numeric_stats.csv"),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    conclusions = []
    if total:
        rate = matched_mask.sum() / total * 100.0
        conclusions.append(f"- Taxa de objetos com match Gaia: {rate:.2f}% ({int(matched_mask.sum())}/{total}).")
    if summary["total_ambiguous"]:
        conclusions.append(f"- Ha {summary['total_ambiguous']} matches ambiguos; estes objetos devem ser revisados antes de conclusoes finais.")
    high_pm = int(df.get("gaia_high_pm_flag", pd.Series(dtype=bool)).fillna(False).sum())
    if high_pm:
        conclusions.append(f"- {high_pm} objetos possuem movimento proprio Gaia elevado e podem ser sensiveis a diferencas de epoca.")
    if not conclusions:
        conclusions.append("- Nao ha dados Gaia suficientes para conclusoes quantitativas.")

    report_lines = [
        "# Relatorio Gaia DR3 x SDSS Local",
        "",
        "## Resumo",
        "",
        f"- Objetos locais: {summary['total_local_objects']}",
        f"- Objetos com RA/DEC: {summary['objects_with_radec']}",
        f"- Objetos com match Gaia: {summary['total_with_match']}",
        f"- Sem match: {summary['total_no_match']}",
        f"- Ambiguos: {summary['total_ambiguous']}",
        f"- Erros: {summary['total_errors']}",
        "",
        "## Cross-match",
        "",
        _markdown_table(match_status_counts),
        "",
        "## Cobertura Gaia",
        "",
        _markdown_table(coverage),
        "",
        "## Estatisticas Numericas",
        "",
        _markdown_table(stats) if not stats.empty else "Sem parametros numericos disponiveis.",
        "",
        "## Subclasses",
        "",
        _markdown_table(spectral_subclass_counts, max_rows=50),
        "",
        "## Classes Espectrais Amplas",
        "",
        _markdown_table(spectral_major_counts),
        "",
        "## Classes Evolutivas/Fisicas Derivadas",
        "",
        _markdown_table(stellar_evolution_counts),
        "",
        "## Qualidade Da Paralaxe",
        "",
        _markdown_table(parallax_quality_counts),
        "",
        "## Matriz Classe Espectral x Classe Evolutiva",
        "",
        _markdown_table(spectral_evolution_matrix.reset_index()) if not spectral_evolution_matrix.empty else "Sem dados comparaveis.",
        "",
        "## Matriz Subclasse SDSS x Classe Evolutiva",
        "",
        _markdown_table(subclass_evolution_matrix.reset_index()) if not subclass_evolution_matrix.empty else "Sem dados comparaveis.",
        "",
        "## Matriz Pipeline x Gaia Teff",
        "",
        _markdown_table(comparison_matrix.reset_index()) if not comparison_matrix.empty else "Sem classes estelares comparaveis por Teff.",
        "",
        "## Matriz Classe SDSS x Gaia DSC",
        "",
        _markdown_table(dsc_matrix.reset_index()) if not dsc_matrix.empty else "Sem probabilidades DSC Gaia comparaveis.",
        "",
        "## Qualidade e Limitacoes",
        "",
        "- O cross-match e posicional e usa coordenadas locais SDSS contra Gaia DR3.",
        "- As classes derivadas por Teff sao derivadas, nao classificacoes espectrais observacionais.",
        "- A classe evolutiva/fisica tambem e derivada e deve ser interpretada como anotacao auxiliar.",
        "- Diferencas de epoca e movimento proprio podem afetar objetos estelares de alto movimento proprio.",
        "- Valores ausentes do Gaia foram preservados como ausentes.",
        "",
        "## Conclusoes Automaticas",
        "",
        "\n".join(conclusions),
        "",
    ]
    (reports_dir / "gaia_sdss_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary
