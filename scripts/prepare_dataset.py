"""
Prepara o dataset partilhado em ``<raiz TCC>/dataset/``.

Uso (na pasta do repositório TCC):
  python scripts/prepare_dataset.py              # só materializa a partir de raw existente
  python scripts/prepare_dataset.py --download # download SDSS + materializar
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.pipelines.download import run as download_run  # noqa: E402
from shared.pipelines.preprocess import run as preprocess_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset astro partilhado (SDSS)")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Descarrega imagens para dataset/raw antes de materializar train/val/test.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Pasta raiz do dataset (omissão: <TCC>/dataset).",
    )
    args = parser.parse_args()

    if args.download:
        download_run(data_root=args.data_root)
    preprocess_run(data_root=args.data_root)
    print("\nConcluído.")


if __name__ == "__main__":
    main()
