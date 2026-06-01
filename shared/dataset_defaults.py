"""Constantes do conjunto astro (pastas = nomes no ImageFolder, ordem alfabética nos índices)."""

# Pastas sob raw/ e train|val|test/
IMAGE_CLASS_FOLDERS = ("star", "galaxy", "quasar")

# Ordem dos índices 0..2 usada pelo torchvision.datasets.ImageFolder (sorted)
IMAGE_FOLDER_CLASS_ORDER = tuple(sorted(IMAGE_CLASS_FOLDERS))

SDSS_DR = "dr19"

# download: nome_pasta -> (valor SQL class, índice legado opcional)
SDSS_CLASS_SQL = {
    "star": "STAR",
    "galaxy": "GALAXY",
    "quasar": "QSO",
}

DEFAULT_TOP_N = 3000

DEFAULT_SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
