from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaia_sdss.object_dataset import build_star_object_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera dataset derivado por objeto STAR sem baixar ou reprocessar SDSS."
    )
    parser.add_argument(
        "--comparison-csv",
        default="dataset/gaia_sdss/gaia_sdss_comparison.csv",
        help="CSV consolidado Gaia x SDSS.",
    )
    parser.add_argument(
        "--output",
        default="dataset/star_objects",
        help="Diretorio de saida da estrutura STAR_000001/.",
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Raiz do dataset local com imagens, espectros, plots e metadados.",
    )
    parser.add_argument(
        "--copy-assets",
        action="store_true",
        help="Copia imagem RGB e plot espectral para a pasta de cada objeto.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limita objetos para teste.")
    return parser.parse_args()


def resolve_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    comparison_csv = resolve_path(args.comparison_csv, root)
    output_dir = resolve_path(args.output, root)
    dataset_root = resolve_path(args.dataset_root, root)

    index = build_star_object_dataset(
        comparison_csv=comparison_csv,
        output_dir=output_dir,
        dataset_root=dataset_root,
        copy_assets=args.copy_assets,
        limit=args.limit,
    )

    print(f"Objetos STAR exportados: {len(index)}")
    print(f"Dataset por objeto: {output_dir}")
    print(f"Indice: {output_dir / 'star_objects_index.csv'}")
    print("Nenhum download SDSS foi executado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
