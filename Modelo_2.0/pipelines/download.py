import os
import csv
import time
import requests

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DR = "dr19"

SQL_URL = f"https://skyserver.sdss.org/{DR}/SkyServerWS/SearchTools/SqlSearch"

CLASSES = {
    "star": ("STAR", 0),
    "galaxy": ("GALAXY", 1),
    "quasar": ("QSO", 2),
}

TOP_N = 3000
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata.csv")


# =========================
# CRIAR DIRETÓRIOS
# =========================
def create_dirs():
    for cls in CLASSES:
        os.makedirs(os.path.join(RAW_DIR, cls), exist_ok=True)


# =========================
# QUERY SDSS (COM RETRY)
# =========================
def run_query(query, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(SQL_URL, params={"cmd": query, "format": "json"}, timeout=20)
            resp.raise_for_status()

            data = resp.json()

            if len(data) == 0:
                return []

            return data[0]["Rows"]

        except Exception as e:
            print(f"[QUERY] tentativa {attempt+1} falhou")

            if attempt == retries - 1:
                print("[QUERY] falhou definitivamente")
                return []

            time.sleep(2)


# =========================
# DOWNLOAD IMAGEM (ROBUSTO)
# =========================
def download_image(ra, dec, retries=3):
    url = (
        f"https://skyserver.sdss.org/{DR}/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra}&dec={dec}&scale=0.4&width=256&height=256"
    )

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)

            if resp.status_code == 200:
                return resp.content

        except Exception:
            print(f"[IMG] tentativa {attempt+1} falhou")

        time.sleep(1)

    return None


# =========================
# PROCESSAR CLASSE
# =========================
def process_class(cls_name, cls_sql):
    print(f"\n🔭 Baixando: {cls_name}")

    cls_path = os.path.join(RAW_DIR, cls_name)

    query = f"""
    SELECT TOP {TOP_N}
        p.ra, p.dec
    FROM PhotoObj p
    JOIN SpecObj s ON p.objID = s.bestObjID
    WHERE s.class = '{cls_sql}'
    """

    rows = run_query(query)

    print(f"Total retornado: {len(rows)}")

    success = 0
    metadata_rows = []

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
            metadata_rows.append({
                "class_name": cls_name,
                "base_name": f"{cls_name}_{i}",
                "image_filename": f"{cls_name}_{i}.jpg",
                "ra": ra,
                "dec": dec,
            })

            if success % 100 == 0:
                print(f"{success} objetos baixados")

            time.sleep(0.2)

        except Exception as e:
            print(f"[ERRO] {cls_name} {i}: {e}")

    print(f"✅ {cls_name}: {success} objetos processados")
    return metadata_rows


def write_metadata(rows):
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    fields = ["class_name", "base_name", "image_filename", "ra", "dec"]
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =========================
# MAIN
# =========================
def run():
    print("\n📥 Iniciando download completo...\n")

    create_dirs()

    all_metadata = []
    for cls_name, (cls_sql, _) in CLASSES.items():
        class_rows = process_class(cls_name, cls_sql)
        all_metadata.extend(class_rows)

    write_metadata(all_metadata)
    print(f"🧾 Metadata salva em: {METADATA_PATH}")

    print("\n✅ Download finalizado!")


if __name__ == "__main__":
    run()
