"""
Inspeciona e visualiza espectros SDSS em arquivos FITS.

Exemplos:
  python scripts/inspect_sdss_spectrum.py dataset/spectra/galaxy/430194949951088640.fits
  python scripts/inspect_sdss_spectrum.py dataset/spectra/galaxy/430194949951088640.fits --plot
  python scripts/inspect_sdss_spectrum.py dataset/spectra/galaxy/430194949951088640.fits --export-csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mostra estrutura, metadados e dados flux/wavelength de um FITS SDSS."
    )
    parser.add_argument("fits_path", help="Arquivo .fits do espectro SDSS.")
    parser.add_argument("--head", type=int, default=8, help="Numero de linhas para exibir.")
    parser.add_argument("--plot", action="store_true", help="Gera um PNG com o espectro.")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Exporta wavelength, flux, model, ivar e sky para CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/plots",
        help="Diretorio para PNG/CSV gerados. Padrao: dataset/plots.",
    )
    return parser.parse_args()


def find_spectrum_hdu(hdul: fits.HDUList) -> fits.BinTableHDU:
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        names = getattr(data, "names", None)
        if names and {"flux", "loglam"}.issubset({name.lower() for name in names}):
            return hdu
    raise ValueError("Nao encontrei uma tabela com colunas flux e loglam.")


def find_metadata_hdu(hdul: fits.HDUList) -> Optional[fits.BinTableHDU]:
    for name in ("SPECOBJ", "SPALL"):
        if name in hdul:
            return hdul[name]
    return None


def column(data, name: str):
    names = {column_name.lower(): column_name for column_name in data.names}
    if name.lower() not in names:
        return None
    return data[names[name.lower()]]


def read_spectrum(fits_path: Path) -> tuple[np.ndarray, dict, list[str]]:
    with fits.open(fits_path) as hdul:
        print("\nEstrutura do FITS:")
        hdul.info()

        spectrum_hdu = find_spectrum_hdu(hdul)
        data = spectrum_hdu.data
        flux = np.asarray(column(data, "flux"), dtype=float)
        loglam = np.asarray(column(data, "loglam"), dtype=float)
        wavelength = np.power(10.0, loglam)

        table = {
            "wavelength": wavelength,
            "flux": flux,
        }
        for optional in ("model", "ivar", "sky", "wdisp"):
            values = column(data, optional)
            if values is not None:
                table[optional] = np.asarray(values, dtype=float)

        metadata = {}
        metadata_hdu = find_metadata_hdu(hdul)
        if metadata_hdu is not None and len(metadata_hdu.data) > 0:
            metadata_names = {name.lower(): name for name in metadata_hdu.data.names}
            for key in ("class", "subclass", "z", "z_err", "plate", "mjd", "fiberid", "run2d"):
                original = metadata_names.get(key)
                if original is not None:
                    value = metadata_hdu.data[original][0]
                    if hasattr(value, "item"):
                        value = value.item()
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="ignore").strip()
                    metadata[key] = value

        return wavelength, table, [str(item) for item in metadata.items()]


def print_summary(wavelength: np.ndarray, table: dict, metadata_items: list[str], head: int) -> None:
    flux = table["flux"]
    print("\nResumo:")
    print(f"pixels: {len(flux)}")
    print(f"wavelength min/max: {np.nanmin(wavelength):.2f} / {np.nanmax(wavelength):.2f} Angstrom")
    print(f"flux min/median/max: {np.nanmin(flux):.5g} / {np.nanmedian(flux):.5g} / {np.nanmax(flux):.5g}")

    if metadata_items:
        print("\nMetadados principais:")
        for item in metadata_items:
            print(f"  {item}")

    shown_columns = [name for name in ("wavelength", "flux", "model", "ivar", "sky") if name in table]
    print(f"\nPrimeiras {min(head, len(flux))} linhas:")
    print(", ".join(shown_columns))
    for i in range(min(head, len(flux))):
        values = [table[name][i] for name in shown_columns]
        print(", ".join(f"{value:.8g}" for value in values))


def save_plot(fits_path: Path, output_dir: Path, table: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{fits_path.stem}_spectrum.png"

    plt.figure(figsize=(12, 5))
    plt.plot(table["wavelength"], table["flux"], linewidth=0.8, label="flux")
    if "model" in table:
        plt.plot(table["wavelength"], table["model"], linewidth=0.8, alpha=0.85, label="model")
    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel("Flux")
    plt.title(f"SDSS spectrum: {fits_path.stem}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def export_csv(fits_path: Path, output_dir: Path, table: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{fits_path.stem}_spectrum.csv"
    columns = list(table.keys())

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i in range(len(table["flux"])):
            writer.writerow([table[name][i] for name in columns])
    return output_path


def main() -> int:
    args = parse_args()
    fits_path = Path(args.fits_path)
    output_dir = Path(args.output_dir)

    if not fits_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {fits_path}")

    wavelength, table, metadata_items = read_spectrum(fits_path)
    print_summary(wavelength, table, metadata_items, head=args.head)

    if args.plot:
        output_path = save_plot(fits_path, output_dir, table)
        print(f"\nGrafico salvo em: {output_path}")

    if args.export_csv:
        output_path = export_csv(fits_path, output_dir, table)
        print(f"CSV salvo em: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
