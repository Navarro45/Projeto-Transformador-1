import requests
import os
from time import sleep

# =========================
# CONFIG
# =========================
dr = "dr19"
BASE_DIR = "data/raw"

CLASSES = {
    "estrelas": {
        "table": "Star",
        "query": """
        SELECT TOP 1000 ra, dec
        FROM Star
        WHERE
        ((flags_r & 0x10000000) != 0)
        AND ((flags_r & 0x8100000c00a4) = 0)
        AND (((flags_r & 0x400000000000) = 0) or (psfmagerr_r <= 0.2))
        AND (((flags_r & 0x100000000000) = 0) or (flags_r & 0x1000) = 0)
        """
    },

    "galaxias": {
        "table": "Galaxy",
        "query": """
        SELECT TOP 1000 ra, dec
        FROM Galaxy
        """
    },

    "quasares": {
        "table": "SpecObj",
        "query": """
        SELECT TOP 1000 ra, dec
        FROM SpecObj
        WHERE class = 'QSO'
        """
    }
}

SQL_URL = f"https://skyserver.sdss.org/{dr}/SkyServerWS/SearchTools/SqlSearch"


# =========================
# FUNÇÕES
# =========================

def create_dirs():
    for cls in CLASSES.keys():
        path = os.path.join(BASE_DIR, cls)
        os.makedirs(path, exist_ok=True)


def fetch_coordinates(query):
    resp = requests.get(SQL_URL, params={"cmd": query, "format": "json"})
    resp.raise_for_status()

    data = resp.json()

    if len(data) == 0:
        return []

    return data[0]["Rows"]


def download_image(ra, dec):
    url = (
        f"https://skyserver.sdss.org/{dr}/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra}&dec={dec}&scale=0.4&width=256&height=256"
    )

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    return resp.content


# =========================
# MAIN LOOP
# =========================

def main():
    create_dirs()

    for class_name, config in CLASSES.items():
        print(f"\n🔭 Baixando classe: {class_name}")

        rows = fetch_coordinates(config["query"])
        print(f"Total encontrado: {len(rows)}")

        save_dir = os.path.join(BASE_DIR, class_name)

        success = 0

        for idx, row in enumerate(rows):
            ra = row["ra"]
            dec = row["dec"]

            filename = f"{class_name}_{idx}_ra{ra:.4f}_dec{dec:.4f}.jpg"
            filepath = os.path.join(save_dir, filename)

            if os.path.exists(filepath):
                continue

            try:
                img = download_image(ra, dec)

                with open(filepath, "wb") as f:
                    f.write(img)

                success += 1

                if success % 50 == 0:
                    print(f"{success} imagens baixadas...")

                # Evita sobrecarregar servidor
                sleep(0.1)

            except Exception as e:
                print(f"Erro [{class_name}] RA={ra} DEC={dec}: {e}")

        print(f"✔ {class_name}: {success} imagens salvas")


if __name__ == "__main__":
    main()