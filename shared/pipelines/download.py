import csv
import os
import time
from typing import Optional

import requests

from shared.dataset_defaults import DEFAULT_TOP_N, SDSS_CLASS_SQL, SDSS_DR
from shared.paths import dataset_layout, default_dataset_root

SQL_URL = f"https://skyserver.sdss.org/{SDSS_DR}/SkyServerWS/SearchTools/SqlSearch"


def _img_url(ra: float, dec: float) -> str:
    return (
        f"https://skyserver.sdss.org/{SDSS_DR}/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra}&dec={dec}&scale=0.4&width=256&height=256"
    )


def create_dirs(raw_dir: str, class_names: tuple) -> None:
    for cls in class_names:
        os.makedirs(os.path.join(raw_dir, cls), exist_ok=True)


def run_query(query: str, retries: int = 3):
    for attempt in range(retries):
        try:
            resp = requests.get(SQL_URL, params={"cmd": query, "format": "json"}, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if len(data) == 0:
                return []
            return data[0]["Rows"]
        except Exception:
            print(f"[QUERY] tentativa {attempt + 1} falhou")
            if attempt == retries - 1:
                print("[QUERY] falhou definitivamente")
                return []
            time.sleep(2)


def download_image(ra: float, dec: float, retries: int = 3):
    url = _img_url(ra, dec)
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.content
        except Exception:
            print(f"[IMG] tentativa {attempt + 1} falhou")
        time.sleep(1)
    return None


def process_class(cls_name: str, cls_sql: str, raw_dir: str, top_n: int, metadata_rows: list):
    print(f"\n Baixando: {cls_name}")

    cls_path = os.path.join(raw_dir, cls_name)
    query = f"""
    SELECT TOP {top_n}
        p.ra, p.dec
    FROM PhotoObj p
    JOIN SpecObj s ON p.objID = s.bestObjID
    WHERE s.class = '{cls_sql}'
    """

    rows = run_query(query)
    print(f"Total retornado: {len(rows)}")

    success = 0
    for i, row in enumerate(rows):
        ra = row["ra"]
        dec = row["dec"]
        img_path = os.path.join(cls_path, f"{cls_name}_{i}.jpg")

        try:
            if not os.path.exists(img_path):
                img = download_image(ra, dec)
                if img is None:
                    continue
                with open(img_path, "wb") as f:
                    f.write(img)

            success += 1
            metadata_rows.append(
                {
                    "class_name": cls_name,
                    "base_name": f"{cls_name}_{i}",
                    "image_filename": f"{cls_name}_{i}.jpg",
                    "ra": ra,
                    "dec": dec,
                }
            )

            if success % 100 == 0:
                print(f"{success} objetos baixados")

            time.sleep(0.2)
        except Exception as e:
            print(f"[ERRO] {cls_name} {i}: {e}")

    print(f" {cls_name}: {success} objetos processados")


def write_metadata(metadata_path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    fields = ["class_name", "base_name", "image_filename", "ra", "dec"]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(data_root: Optional[str] = None, top_n: Optional[int] = None) -> None:
    """
    Descarrega JPEGs do SDSS para ``<data_root>/raw/<classe>/``.
    ``data_root`` por omissão: ``<repo>/dataset``.
    """
    paths = dataset_layout(data_root or default_dataset_root())
    raw_dir = paths["raw"]
    top_n = DEFAULT_TOP_N if top_n is None else top_n

    print("\n Iniciando download completo (dataset partilhado)...\n")

    create_dirs(raw_dir, tuple(SDSS_CLASS_SQL.keys()))

    all_metadata: list = []
    for cls_name, cls_sql in SDSS_CLASS_SQL.items():
        process_class(cls_name, cls_sql, raw_dir, top_n, all_metadata)

    write_metadata(paths["metadata"], all_metadata)
    print(f" Metadata salva em: {paths['metadata']}")
    print("\n Download finalizado.")


if __name__ == "__main__":
    run()
