"""
Baixa imagens e espectros SDSS DR20 a partir de um CSV do SkyServer.

Uso basico, na raiz do projeto:
  python scripts/download_sdss_dr20_dataset.py --csv MyResult_2026824.csv --limit 10
  python scripts/download_sdss_dr20_dataset.py --csv MyResult_2026824.csv

Saida:
  dataset/images/<classe>/<objID>.jpg
  dataset/spectra/<classe>/<specObjID>.fits
  dataset/plots/<classe>/<specObjID>_spectrum.png
  dataset/metadata/spectra/<classe>/<specObjID>.json
  dataset/reports/subclass_counts.csv
  dataset/reports/subclass_report.md
  dataset/manifests/download_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlencode

import numpy as np
import requests


SDSS_DR = "dr20"
SKYSERVER_BASE = f"https://skyserver.sdss.org/{SDSS_DR}"
IMG_CUTOUT_URL = f"{SKYSERVER_BASE}/SkyServerWS/ImgCutout/getjpeg"
SAS_DR20_BASE = "https://dr20.sdss.org/sas/dr20"

CLASS_DIRS = {
    "GALAXY": "galaxy",
    "QSO": "quasar",
    "STAR": "star",
}

MANIFEST_FIELDS = [
    "source_index",
    "objID",
    "specObjID",
    "spectralClass",
    "ra",
    "dec",
    "redshift",
    "plate",
    "mjd",
    "fiber",
    "run2d",
    "subclass",
    "spectral_pixels",
    "wavelength_min",
    "wavelength_max",
    "flux_min",
    "flux_median",
    "flux_max",
    "image_path",
    "spectrum_path",
    "spectrum_csv_path",
    "spectrum_plot_path",
    "spectrum_metadata_path",
    "image_status",
    "spectrum_status",
    "spectrum_processing_status",
    "spectrum_url",
    "error",
]


@dataclass(frozen=True)
class SpectrumId:
    plate: int
    mjd: int
    fiber: int
    run2d: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa JPEGs e FITS espectrais do SDSS DR20 para a pasta dataset/."
    )
    parser.add_argument(
        "--csv",
        default="MyResult_2026824.csv",
        help="CSV de entrada com objID, ra, dec, specObjID e spectralClass.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Diretorio de saida. Padrao: dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Baixa no maximo N linhas, util para testar antes do download completo.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=sorted(CLASS_DIRS),
        default=None,
        help="Filtra classes espectrais. Ex.: --classes GALAXY QSO.",
    )
    parser.add_argument("--skip-images", action="store_true", help="Nao baixa JPEGs.")
    parser.add_argument("--skip-spectra", action="store_true", help="Nao baixa FITS.")
    parser.add_argument(
        "--skip-spectrum-processing",
        action="store_true",
        help="Nao gera plots, metadados e relatorios a partir dos FITS.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Gera metadados e relatorio, mas nao salva PNG dos espectros.",
    )
    parser.add_argument(
        "--save-spectrum-csv",
        action="store_true",
        help="Tambem extrai wavelength/flux dos FITS para CSV em dataset/spectra_csv/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Baixa novamente arquivos que ja existem.",
    )
    parser.add_argument("--width", type=int, default=256, help="Largura do JPEG.")
    parser.add_argument("--height", type=int, default=256, help="Altura do JPEG.")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.4,
        help="Escala do JPEG em arcsec/pixel.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout HTTP.")
    parser.add_argument("--retries", type=int, default=3, help="Tentativas por arquivo.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Pausa entre downloads, em segundos.",
    )
    return parser.parse_args()


def decode_run2d(value: int) -> str:
    """Converte o campo run2d codificado no specObjID para o nome usado no SAS."""
    if 1000 <= value < 10000:
        major = value // 100
        patch = value % 100
        return f"v5_{major}_{patch}"
    if value >= 10000:
        n = 5 + (value // 10000)
        mp = value % 10000
        m = mp // 100
        p = mp % 100
        return f"v{n}_{m}_{p}"
    return str(value)


def json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def fits_column(data, name: str):
    names = {column_name.lower(): column_name for column_name in data.names}
    if name.lower() not in names:
        return None
    return data[names[name.lower()]]


def read_spectrum_product(fits_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    try:
        from astropy.io import fits
    except ImportError as error:
        raise ImportError("astropy e necessario para ler e processar espectros FITS.") from error

    with fits.open(fits_path) as hdul:
        spectrum_hdu = None
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            names = getattr(data, "names", None)
            if names and {"loglam", "flux"}.issubset({name.lower() for name in names}):
                spectrum_hdu = hdu
                break
        if spectrum_hdu is None:
            raise ValueError(f"FITS sem tabela espectral loglam/flux: {fits_path}")

        data = spectrum_hdu.data
        loglam = np.asarray(fits_column(data, "loglam"), dtype=float)
        flux = np.asarray(fits_column(data, "flux"), dtype=float)
        wavelength = np.power(10.0, loglam)

        table = {
            "wavelength": wavelength,
            "flux": flux,
        }
        for optional in ("model", "ivar", "sky", "wdisp"):
            values = fits_column(data, optional)
            if values is not None:
                table[optional] = np.asarray(values, dtype=float)

        metadata = {
            "fits_path": str(fits_path),
            "spectrum_hdu": spectrum_hdu.name,
            "spectral_pixels": int(len(flux)),
            "wavelength_min": float(np.nanmin(wavelength)),
            "wavelength_max": float(np.nanmax(wavelength)),
            "flux_min": float(np.nanmin(flux)),
            "flux_median": float(np.nanmedian(flux)),
            "flux_max": float(np.nanmax(flux)),
            "extensions": [hdu.name for hdu in hdul],
        }

        for hdu_name in ("SPECOBJ", "SPALL"):
            if hdu_name not in hdul or len(hdul[hdu_name].data) == 0:
                continue

            meta_data = hdul[hdu_name].data
            metadata["metadata_hdu"] = hdu_name
            metadata_names = {name.lower(): name for name in meta_data.names}
            for key in (
                "survey",
                "instrument",
                "class",
                "subclass",
                "z",
                "z_err",
                "plate",
                "mjd",
                "fiberid",
                "run2d",
                "sn_median_all",
            ):
                original = metadata_names.get(key)
                if original is not None:
                    metadata[key] = json_safe(meta_data[original][0])
            break

        return table, metadata


def save_spectrum_plot(
    table: dict[str, np.ndarray],
    plot_path: Path,
    title: str,
    overwrite: bool,
) -> str:
    if plot_path.exists() and not overwrite:
        return "exists"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = plot_path.with_suffix(plot_path.suffix + ".part.png")

    plt.figure(figsize=(12, 5))
    plt.plot(table["wavelength"], table["flux"], linewidth=0.8, label="flux")
    if "model" in table:
        plt.plot(table["wavelength"], table["model"], linewidth=0.8, alpha=0.85, label="model")
    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel("Flux")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(tmp_path, dpi=160)
    plt.close()
    tmp_path.replace(plot_path)
    return "written"


def save_spectrum_metadata(metadata: dict, metadata_path: Path, overwrite: bool) -> str:
    if metadata_path.exists() and not overwrite:
        return "exists"

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".part")
    tmp_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(metadata_path)
    return "written"


def decode_specobjid(spec_obj_id: str) -> SpectrumId:
    """Decodifica specObjID CAS-style em plate, mjd, fiber e run2d.

    A convencao moderna usa:
      bits 50-63: plate
      bits 38-49: fiber
      bits 24-37: MJD - 50000
      bits 10-23: run2d
    """
    sid = int(spec_obj_id)
    plate = sid >> 50
    fiber = (sid >> 38) & 0x0FFF
    mjd = ((sid >> 24) & 0x3FFF) + 50000
    run2d = decode_run2d((sid >> 10) & 0x3FFF)

    if plate <= 0 or fiber <= 0 or mjd <= 50000:
        raise ValueError(f"specObjID invalido ou nao suportado: {spec_obj_id}")

    return SpectrumId(plate=plate, mjd=mjd, fiber=fiber, run2d=run2d)


def image_url(ra: str, dec: str, width: int, height: int, scale: float) -> str:
    params = {
        "ra": ra,
        "dec": dec,
        "scale": scale,
        "width": width,
        "height": height,
    }
    return f"{IMG_CUTOUT_URL}?{urlencode(params)}"


def spectrum_url_candidates(spectrum_id: SpectrumId) -> list[str]:
    plate = spectrum_id.plate
    mjd = spectrum_id.mjd
    fiber = spectrum_id.fiber
    run2d = spectrum_id.run2d
    filename = f"spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits"

    if run2d == "v5_13_2":
        rel_roots = [
            f"prior-surveys/sdss4-dr17-eboss/spectro/redux/{run2d}/spectra/lite/{plate:04d}",
            f"prior-surveys/sdss4-dr17-eboss/spectro/redux/{run2d}/spectra/full/{plate:04d}",
            # Mantido como fallback porque a pagina DR20 aponta parte dos dados
            # herdados para o espelho DR18.
            f"https://dr18.sdss.org/sas/dr18/prior-surveys/sdss4-dr17-eboss/spectro/redux/{run2d}/spectra/lite/{plate:04d}",
            f"https://dr18.sdss.org/sas/dr18/prior-surveys/sdss4-dr17-eboss/spectro/redux/{run2d}/spectra/full/{plate:04d}",
        ]
    elif run2d in {"26", "103", "104"}:
        rel_roots = [
            f"prior-surveys/sdss2-dr8-sdss/spectro/redux/{run2d}/spectra/lite/{plate:04d}",
            f"prior-surveys/sdss2-dr8-sdss/spectro/redux/{run2d}/spectra/{plate:04d}",
            f"https://dr18.sdss.org/sas/dr18/prior-surveys/sdss2-dr8-sdss/spectro/redux/{run2d}/spectra/lite/{plate:04d}",
            f"https://dr18.sdss.org/sas/dr18/prior-surveys/sdss2-dr8-sdss/spectro/redux/{run2d}/spectra/{plate:04d}",
        ]
    else:
        rel_roots = [
            f"spectro/boss/redux/{run2d}/spectra/lite/{plate:04d}",
            f"spectro/boss/redux/{run2d}/spectra/full/{plate:04d}",
        ]

    urls = []
    for root in rel_roots:
        if root.startswith("https://"):
            urls.append(f"{root}/{filename}")
        else:
            urls.append(f"{SAS_DR20_BASE}/{root}/{filename}")
    return urls


def request_bytes(
    session: requests.Session,
    url: str,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.HTTPError as error:
            last_error = error
            status = error.response.status_code if error.response is not None else None
            if status == 404:
                break
        except requests.RequestException as error:
            last_error = error

        if attempt < retries:
            time.sleep(sleep_seconds)

    if last_error is None:
        raise RuntimeError(f"download falhou: {url}")
    raise last_error


def download_to_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    timeout: float,
    retries: int,
    sleep_seconds: float,
    overwrite: bool,
) -> str:
    if output_path.exists() and not overwrite:
        return "exists"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    content = request_bytes(session, url, timeout, retries, sleep_seconds)
    tmp_path.write_bytes(content)
    tmp_path.replace(output_path)
    return "downloaded"


def download_first_spectrum_candidate(
    session: requests.Session,
    urls: Iterable[str],
    output_path: Path,
    timeout: float,
    retries: int,
    sleep_seconds: float,
    overwrite: bool,
) -> tuple[str, str]:
    if output_path.exists() and not overwrite:
        return "exists", ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for url in urls:
        try:
            status = download_to_file(
                session,
                url,
                output_path,
                timeout=timeout,
                retries=retries,
                sleep_seconds=sleep_seconds,
                overwrite=True,
            )
            return status, url
        except requests.HTTPError as error:
            status_code = error.response.status_code if error.response is not None else "?"
            last_error = f"HTTP {status_code}: {url}"
            if status_code != 404:
                break
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
            break

    raise RuntimeError(last_error or "nenhuma URL candidata funcionou")


def extract_spectrum_csv(fits_path: Path, csv_path: Path, overwrite: bool) -> str:
    if csv_path.exists() and not overwrite:
        return "exists"

    try:
        import numpy as np
        from astropy.io import fits
    except ImportError as error:
        raise ImportError(
            "Para --save-spectrum-csv instale astropy e numpy."
        ) from error

    with fits.open(fits_path) as hdul:
        table = None
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            names = getattr(data, "names", None)
            if names and {"loglam", "flux"}.issubset({name.lower() for name in names}):
                table = data
                break
        if table is None:
            raise ValueError(f"FITS sem colunas loglam/flux: {fits_path}")

        name_map = {name.lower(): name for name in table.names}
        wavelength = np.power(10.0, np.asarray(table[name_map["loglam"]], dtype=float))
        flux = np.asarray(table[name_map["flux"]], dtype=float)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".part")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength", "flux"])
        writer.writerows(zip(wavelength, flux))
    tmp_path.replace(csv_path)
    return "written"


def copy_source_csv(source_csv: Path, dataset_dir: Path) -> None:
    target = dataset_dir / "manifests" / source_csv.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_csv.resolve() != target.resolve():
        shutil.copy2(source_csv, target)


def normalized_class(value: str) -> str:
    return CLASS_DIRS.get(value.strip().upper(), value.strip().lower() or "unknown")


def iter_rows(csv_path: Path, allowed_classes: Optional[set[str]], limit: Optional[int]):
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        emitted = 0
        for source_index, row in enumerate(reader):
            spectral_class = row.get("spectralClass", "").strip().upper()
            if allowed_classes is not None and spectral_class not in allowed_classes:
                continue
            yield source_index, row
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def write_manifest_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()


def append_manifest_row(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writerow(row)


def process_spectrum_outputs(
    fits_path: Path,
    plot_path: Path,
    metadata_path: Path,
    args: argparse.Namespace,
    context: dict,
) -> tuple[str, dict]:
    table, metadata = read_spectrum_product(fits_path)
    metadata.update(context)

    metadata_status = save_spectrum_metadata(
        metadata,
        metadata_path,
        overwrite=args.overwrite,
    )

    if args.skip_plots:
        plot_status = "skipped"
    else:
        plot_status = save_spectrum_plot(
            table,
            plot_path,
            title=f"SDSS spectrum: {context.get('specObjID', fits_path.stem)}",
            overwrite=args.overwrite,
        )

    return f"metadata_{metadata_status};plot_{plot_status}", metadata


def load_all_spectrum_metadata(dataset_dir: Path) -> list[dict]:
    metadata_root = dataset_dir / "metadata" / "spectra"
    records = []
    if not metadata_root.exists():
        return records

    for metadata_path in sorted(metadata_root.rglob("*.json")):
        try:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        record["spectrum_metadata_path"] = str(metadata_path)
        if "spectralClass" not in record:
            record["spectralClass"] = metadata_path.parent.name.upper()
        if "subclass" not in record or str(record["subclass"]).strip() == "":
            record["subclass"] = "SEM_SUBCLASSE"
        records.append(record)
    return records


def write_spectrum_metadata_index(dataset_dir: Path, records: list[dict]) -> Path:
    metadata_dir = dataset_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_path = metadata_dir / "spectra_metadata.csv"
    fields = [
        "objID",
        "specObjID",
        "spectralClass",
        "class",
        "subclass",
        "z",
        "z_err",
        "plate",
        "mjd",
        "fiberid",
        "run2d",
        "spectral_pixels",
        "wavelength_min",
        "wavelength_max",
        "flux_min",
        "flux_median",
        "flux_max",
        "image_path",
        "spectrum_path",
        "spectrum_metadata_path",
    ]

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    return index_path


def write_subclass_reports(dataset_dir: Path, manifest_rows: list[dict]) -> None:
    reports_dir = dataset_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    processed_rows = load_all_spectrum_metadata(dataset_dir)
    if not processed_rows:
        processed_rows = [
            row for row in manifest_rows
            if str(row.get("spectrum_processing_status", "")).startswith("metadata_")
        ]

    metadata_index_path = write_spectrum_metadata_index(dataset_dir, processed_rows)

    class_counts = Counter(row.get("spectralClass", "UNKNOWN") for row in processed_rows)
    subclass_counts = Counter(
        (
            row.get("spectralClass", "UNKNOWN"),
            row.get("subclass", "") or "SEM_SUBCLASSE",
        )
        for row in processed_rows
    )

    csv_path = reports_dir / "subclass_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["spectralClass", "subclass", "count"],
        )
        writer.writeheader()
        for (spectral_class, subclass), count in sorted(subclass_counts.items()):
            writer.writerow(
                {
                    "spectralClass": spectral_class,
                    "subclass": subclass,
                    "count": count,
                }
            )

    json_path = reports_dir / "subclass_counts.json"
    json_payload = {
        "total_processed_spectra": len(processed_rows),
        "metadata_index": str(metadata_index_path),
        "class_counts": dict(sorted(class_counts.items())),
        "subclass_counts": [
            {
                "spectralClass": spectral_class,
                "subclass": subclass,
                "count": count,
            }
            for (spectral_class, subclass), count in sorted(subclass_counts.items())
        ],
    }
    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_path = reports_dir / "subclass_report.md"
    lines = [
        "# Relatorio de Subclasses Espectrais",
        "",
        f"Total de espectros processados: {len(processed_rows)}",
        f"Indice de metadados: `{metadata_index_path}`",
        "",
        "## Totais por Classe",
        "",
        "| Classe | Quantidade |",
        "|---|---:|",
    ]
    for spectral_class, count in sorted(class_counts.items()):
        lines.append(f"| {spectral_class} | {count} |")

    lines.extend(
        [
            "",
            "## Totais por Subclasse",
            "",
            "| Classe | Subclasse | Quantidade |",
            "|---|---|---:|",
        ]
    )
    for (spectral_class, subclass), count in sorted(subclass_counts.items()):
        lines.append(f"| {spectral_class} | {subclass} | {count} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_row(
    session: requests.Session,
    source_index: int,
    row: dict,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> dict:
    obj_id = row.get("objID", "").strip()
    spec_obj_id = row.get("specObjID", "").strip()
    spectral_class = row.get("spectralClass", "").strip().upper()
    class_dir = normalized_class(spectral_class)

    image_path = dataset_dir / "images" / class_dir / f"{obj_id}.jpg"
    spectrum_path = dataset_dir / "spectra" / class_dir / f"{spec_obj_id}.fits"
    spectrum_csv_path = dataset_dir / "spectra_csv" / class_dir / f"{spec_obj_id}.csv"
    spectrum_plot_path = dataset_dir / "plots" / class_dir / f"{spec_obj_id}_spectrum.png"
    spectrum_metadata_path = dataset_dir / "metadata" / "spectra" / class_dir / f"{spec_obj_id}.json"

    manifest_row = {
        "source_index": source_index,
        "objID": obj_id,
        "specObjID": spec_obj_id,
        "spectralClass": spectral_class,
        "ra": row.get("ra", ""),
        "dec": row.get("dec", ""),
        "redshift": row.get("redshift", ""),
        "image_path": str(image_path),
        "spectrum_path": str(spectrum_path),
        "spectrum_csv_path": str(spectrum_csv_path) if args.save_spectrum_csv else "",
        "spectrum_plot_path": "" if args.skip_plots else str(spectrum_plot_path),
        "spectrum_metadata_path": str(spectrum_metadata_path),
        "image_status": "skipped" if args.skip_images else "",
        "spectrum_status": "skipped" if args.skip_spectra else "",
        "spectrum_processing_status": "skipped" if args.skip_spectrum_processing else "",
        "spectrum_url": "",
        "error": "",
    }

    try:
        spectrum_id = decode_specobjid(spec_obj_id)
        manifest_row.update(
            {
                "plate": spectrum_id.plate,
                "mjd": spectrum_id.mjd,
                "fiber": spectrum_id.fiber,
                "run2d": spectrum_id.run2d,
            }
        )

        if not args.skip_images:
            url = image_url(
                row["ra"],
                row["dec"],
                width=args.width,
                height=args.height,
                scale=args.scale,
            )
            manifest_row["image_status"] = download_to_file(
                session,
                url,
                image_path,
                timeout=args.timeout,
                retries=args.retries,
                sleep_seconds=args.sleep,
                overwrite=args.overwrite,
            )
            time.sleep(args.sleep)

        if not args.skip_spectra:
            status, used_url = download_first_spectrum_candidate(
                session,
                spectrum_url_candidates(spectrum_id),
                spectrum_path,
                timeout=args.timeout,
                retries=args.retries,
                sleep_seconds=args.sleep,
                overwrite=args.overwrite,
            )
            manifest_row["spectrum_status"] = status
            manifest_row["spectrum_url"] = used_url
            time.sleep(args.sleep)

            if args.save_spectrum_csv:
                extract_status = extract_spectrum_csv(
                    spectrum_path,
                    spectrum_csv_path,
                    overwrite=args.overwrite,
                )
                manifest_row["spectrum_status"] = f"{status};csv_{extract_status}"

            if not args.skip_spectrum_processing:
                processing_status, metadata = process_spectrum_outputs(
                    spectrum_path,
                    spectrum_plot_path,
                    spectrum_metadata_path,
                    args,
                    context={
                        "source_index": source_index,
                        "objID": obj_id,
                        "specObjID": spec_obj_id,
                        "spectralClass": spectral_class,
                        "ra": row.get("ra", ""),
                        "dec": row.get("dec", ""),
                        "redshift_csv": row.get("redshift", ""),
                        "plate_decoded": spectrum_id.plate,
                        "mjd_decoded": spectrum_id.mjd,
                        "fiber_decoded": spectrum_id.fiber,
                        "run2d_decoded": spectrum_id.run2d,
                        "image_path": str(image_path),
                        "spectrum_path": str(spectrum_path),
                    },
                )
                manifest_row["spectrum_processing_status"] = processing_status
                manifest_row["subclass"] = str(metadata.get("subclass", "")).strip()
                manifest_row["spectral_pixels"] = metadata.get("spectral_pixels", "")
                manifest_row["wavelength_min"] = metadata.get("wavelength_min", "")
                manifest_row["wavelength_max"] = metadata.get("wavelength_max", "")
                manifest_row["flux_min"] = metadata.get("flux_min", "")
                manifest_row["flux_median"] = metadata.get("flux_median", "")
                manifest_row["flux_max"] = metadata.get("flux_max", "")

    except Exception as error:  # noqa: BLE001
        manifest_row["error"] = f"{type(error).__name__}: {error}"

    return manifest_row


def main() -> int:
    args = parse_args()
    root = repo_root()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir

    if not csv_path.exists():
        print(f"CSV nao encontrado: {csv_path}", file=sys.stderr)
        return 2

    dataset_dir.mkdir(parents=True, exist_ok=True)
    copy_source_csv(csv_path, dataset_dir)

    manifest_path = dataset_dir / "manifests" / "download_manifest.csv"
    write_manifest_header(manifest_path)

    allowed_classes = set(args.classes) if args.classes else None
    counters = {
        "rows": 0,
        "image_ok": 0,
        "spectrum_ok": 0,
        "processed_ok": 0,
        "errors": 0,
    }
    manifest_rows: list[dict] = []

    print(f"CSV: {csv_path}")
    print(f"Saida: {dataset_dir}")
    print(f"Manifesto: {manifest_path}")

    with requests.Session() as session:
        session.headers.update({"User-Agent": "TCC-SDSS-DR20-dataset-downloader/1.0"})
        for source_index, row in iter_rows(csv_path, allowed_classes, args.limit):
            result = process_row(session, source_index, row, dataset_dir, args)
            append_manifest_row(manifest_path, result)
            manifest_rows.append(result)

            counters["rows"] += 1
            if result["image_status"] in {"downloaded", "exists", "skipped"}:
                counters["image_ok"] += 1
            if str(result["spectrum_status"]).startswith(("downloaded", "exists", "skipped")):
                counters["spectrum_ok"] += 1
            if str(result["spectrum_processing_status"]).startswith(("metadata_", "skipped")):
                counters["processed_ok"] += 1
            if result["error"]:
                counters["errors"] += 1

            if counters["rows"] % 25 == 0:
                print(
                    "Processados: {rows} | imagens ok: {image_ok} | "
                    "espectros ok: {spectrum_ok} | pos-processados: {processed_ok} | "
                    "erros: {errors}".format(**counters)
                )

    if not args.skip_spectrum_processing:
        write_subclass_reports(dataset_dir, manifest_rows)

    print(
        "Finalizado: {rows} linhas | imagens ok: {image_ok} | "
        "espectros ok: {spectrum_ok} | pos-processados: {processed_ok} | "
        "erros: {errors}".format(**counters)
    )
    return 0 if counters["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
