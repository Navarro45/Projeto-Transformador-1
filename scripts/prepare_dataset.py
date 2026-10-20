"""Prepara os splits de imagens sem iniciar a pipeline de modelos."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.dataset_defaults import DEFAULT_SPLIT  # noqa: E402
from shared.paths import default_dataset_root  # noqa: E402
from shared.pipelines.download import run as download_run  # noqa: E402
from shared.pipelines.preprocess import run as preprocess_run  # noqa: E402


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else _ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Divide as imagens locais em 70% treino, 15% validacao e 15% teste, "
            "sem executar modelos."
        )
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Raiz do dataset. Padrao: <TCC>/dataset.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Imagens de origem. Padrao: <data-root>/images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed da divisao deterministica.")
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=8,
        help="Quantidade de imagens validadas em paralelo.",
    )
    parser.add_argument(
        "--materialization",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="Hardlink evita duplicar os bytes; copy cria copias independentes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove somente dataset/train, dataset/val e dataset/test antes de recriar.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Opcional: baixa um novo conjunto em <data-root>/raw e o usa como origem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = _resolve_path(args.data_root) or Path(default_dataset_root())
    source_dir = _resolve_path(args.source_dir)

    if args.download:
        download_run(data_root=str(data_root))
        if source_dir is None:
            source_dir = data_root / "raw"

    preprocess_run(
        data_root=str(data_root),
        source_dir=str(source_dir) if source_dir is not None else None,
        split=DEFAULT_SPLIT,
        seed=args.seed,
        materialization=args.materialization,
        overwrite=args.overwrite,
        validation_workers=args.validation_workers,
    )


if __name__ == "__main__":
    main()
