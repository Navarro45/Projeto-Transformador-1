from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SIZE = 256
GRID_NAMES = (
    "superior_esquerda",
    "superior",
    "superior_direita",
    "esquerda",
    "centro",
    "direita",
    "inferior_esquerda",
    "inferior",
    "inferior_direita",
)


@dataclass(frozen=True)
class ExtractionResult:
    foreground: np.ndarray
    alpha: np.ndarray
    bbox: tuple[int, int, int, int]
    background_rgb: tuple[float, float, float]
    signal_to_noise: float
    mask_area: int


def stable_seed(*parts) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def validate_image_size(image: Image.Image, image_size: int = IMAGE_SIZE) -> None:
    if image.size != (image_size, image_size):
        raise ValueError(
            f"Imagem deve possuir {image_size}x{image_size}, recebido {image.size[0]}x{image.size[1]}"
        )


def extract_centered_object(
    image: Image.Image,
    image_size: int = IMAGE_SIZE,
    center_seed_radius: int = 12,
) -> ExtractionResult:
    """Extrai a fonte que cruza a regiao central, nunca a maior fonte global."""
    source = image.convert("RGB")
    validate_image_size(source, image_size)
    rgb = np.asarray(source, dtype=np.float32)
    border_width = max(16, image_size // 8)
    border = np.concatenate(
        [
            rgb[:border_width].reshape(-1, 3),
            rgb[-border_width:].reshape(-1, 3),
            rgb[border_width:-border_width, :border_width].reshape(-1, 3),
            rgb[border_width:-border_width, -border_width:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    residual = np.clip(rgb - background.reshape(1, 1, 3), 0.0, 255.0)
    detection = residual.max(axis=2)
    border_detection = np.concatenate(
        [
            detection[:border_width].ravel(),
            detection[-border_width:].ravel(),
            detection[border_width:-border_width, :border_width].ravel(),
            detection[border_width:-border_width, -border_width:].ravel(),
        ]
    )
    median = float(np.median(border_detection))
    noise = max(1.0, 1.4826 * float(np.median(np.abs(border_detection - median))))

    selected = None
    selected_peak = 0.0
    center = image_size // 2
    yy, xx = np.ogrid[:image_size, :image_size]
    center_seed = (xx - center) ** 2 + (yy - center) ** 2 <= center_seed_radius**2
    kernel = np.ones((3, 3), np.uint8)

    for sigma_factor in (3.0, 2.0, 1.25):
        threshold = max(2.0, median + sigma_factor * noise)
        binary = (detection >= threshold).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        candidates = []
        for label in range(1, labels_count):
            component = labels == label
            overlap = int(np.count_nonzero(component & center_seed))
            cx, cy = centroids[label]
            distance = float(math.hypot(cx - center, cy - center))
            if overlap or distance <= center_seed_radius:
                peak = float(detection[component].max())
                candidates.append((overlap, -distance, peak, label))
        if candidates:
            label = max(candidates)[-1]
            selected = (labels == label).astype(np.uint8)
            selected_peak = float(detection[selected.astype(bool)].max())
            break

    if selected is None:
        raise ValueError("Nao foi encontrada fonte confiavel na regiao central")

    # Impede que pontes de baixo contraste anexem vizinhos ao alvo central.
    # O raio e estimado pela queda do perfil radial e permanece conservador
    # porque os niveis especializados contem apenas fontes estelares.
    central_peak = float(detection[center_seed].max())
    radial_cutoff = max(1.5 * noise, 0.03 * central_peak)
    distances = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    radial_limit = 18
    below_count = 0
    for radius in range(4, 25):
        annulus = (distances >= radius - 1) & (distances < radius)
        percentile = float(np.percentile(detection[annulus], 90))
        if percentile < radial_cutoff:
            below_count += 1
            if below_count >= 2:
                radial_limit = max(6, radius)
                break
        else:
            below_count = 0
    selected = selected * (distances <= radial_limit).astype(np.uint8)
    labels_count, labels, _, centroids = cv2.connectedComponentsWithStats(selected, 8)
    candidates = []
    tight_seed = distances <= 6
    for label in range(1, labels_count):
        component = labels == label
        overlap = int(np.count_nonzero(component & tight_seed))
        cx, cy = centroids[label]
        distance = float(math.hypot(cx - center, cy - center))
        if overlap or distance <= 8:
            candidates.append((overlap, -distance, label))
    if not candidates:
        raise ValueError("Fonte detectada nao permanece conectada ao centro")
    selected = (labels == max(candidates)[-1]).astype(np.uint8)
    selected = cv2.dilate(selected, kernel, iterations=2)
    selected = selected * (distances <= radial_limit + 2).astype(np.uint8)
    alpha = cv2.GaussianBlur(selected.astype(np.float32), (0, 0), sigmaX=1.2)
    if float(alpha.max()) <= 0:
        raise ValueError("Mascara central vazia")
    alpha = np.clip(alpha / float(alpha.max()), 0.0, 1.0)
    support = alpha > 0.02
    ys, xs = np.where(support)
    if len(xs) < 3:
        raise ValueError("Mascara central pequena demais")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if x1 - x0 >= image_size or y1 - y0 >= image_size:
        raise ValueError("Mascara central ocupa toda a imagem")

    signal_to_noise = selected_peak / noise
    if not np.isfinite(signal_to_noise) or signal_to_noise < 1.25:
        raise ValueError("Relacao sinal-ruido insuficiente para extracao")

    foreground = np.clip(residual, 0, 255).astype(np.uint8)
    return ExtractionResult(
        foreground=foreground,
        alpha=alpha,
        bbox=(x0, y0, x1, y1),
        background_rgb=tuple(float(value) for value in background),
        signal_to_noise=float(signal_to_noise),
        mask_area=int(np.count_nonzero(support)),
    )


def generate_synthetic_image(
    image: Image.Image,
    seed: int,
    image_size: int = IMAGE_SIZE,
    safety_margin: int = 4,
) -> tuple[Image.Image, dict, ExtractionResult]:
    extraction = extract_centered_object(image, image_size=image_size)
    x0, y0, x1, y1 = extraction.bbox
    width, height = x1 - x0, y1 - y0
    rng = random.Random(int(seed))
    region_index = rng.randrange(9)
    row, column = divmod(region_index, 3)

    min_center_x = safety_margin + width / 2
    max_center_x = image_size - safety_margin - width / 2
    min_center_y = safety_margin + height / 2
    max_center_y = image_size - safety_margin - height / 2
    if min_center_x > max_center_x or min_center_y > max_center_y:
        raise ValueError("Objeto extraido nao cabe no canvas com a margem configurada")

    cell_x0, cell_x1 = column * image_size / 3, (column + 1) * image_size / 3
    cell_y0, cell_y1 = row * image_size / 3, (row + 1) * image_size / 3
    allowed_x0, allowed_x1 = max(min_center_x, cell_x0), min(max_center_x, cell_x1)
    allowed_y0, allowed_y1 = max(min_center_y, cell_y0), min(max_center_y, cell_y1)
    if allowed_x0 > allowed_x1:
        allowed_x0, allowed_x1 = min_center_x, max_center_x
    if allowed_y0 > allowed_y1:
        allowed_y0, allowed_y1 = min_center_y, max_center_y

    target_center_x = rng.uniform(allowed_x0, allowed_x1)
    target_center_y = rng.uniform(allowed_y0, allowed_y1)
    target_x = int(round(target_center_x - width / 2))
    target_y = int(round(target_center_y - height / 2))
    target_x = min(max(safety_margin, target_x), image_size - safety_margin - width)
    target_y = min(max(safety_margin, target_y), image_size - safety_margin - height)

    crop = extraction.foreground[y0:y1, x0:x1]
    crop_alpha = extraction.alpha[y0:y1, x0:x1, None]
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    destination = canvas[target_y : target_y + height, target_x : target_x + width]
    destination[:] = crop.astype(np.float32) * crop_alpha
    output = np.clip(canvas, 0, 255).astype(np.uint8)

    metadata = {
        "seed": int(seed),
        "grid_region": GRID_NAMES[region_index],
        "position_x": int(target_x),
        "position_y": int(target_y),
        "object_width": int(width),
        "object_height": int(height),
        "mask_area": extraction.mask_area,
        "signal_to_noise": extraction.signal_to_noise,
        "background_r": extraction.background_rgb[0],
        "background_g": extraction.background_rgb[1],
        "background_b": extraction.background_rgb[2],
    }
    return Image.fromarray(output, mode="RGB"), metadata, extraction


def build_balanced_training_records(
    records: pd.DataFrame,
    node_key: str,
    target_per_class: int = 500,
    min_synthetic_fraction: float = 0.10,
    max_synthetic_per_source: int = 10,
    seed: int = 42,
    source_validator: Callable[[str], tuple[bool, str]] | None = None,
) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    if target_per_class < 2:
        raise ValueError("target_per_class deve ser maior ou igual a 2")
    if not 0.0 < min_synthetic_fraction < 1.0:
        raise ValueError("min_synthetic_fraction deve estar entre 0 e 1")
    if max_synthetic_per_source < 1:
        raise ValueError("max_synthetic_per_source deve ser maior ou igual a 1")

    minimum_synthetic = max(1, int(math.ceil(target_per_class * min_synthetic_fraction)))
    maximum_real = target_per_class - minimum_synthetic
    parts: list[pd.DataFrame] = []

    for class_label, group in records.groupby("target_label", sort=False):
        group = group.sort_values(["object_id", "image_path"]).reset_index(drop=True)
        rng = random.Random(stable_seed(seed, node_key, class_label, "selection"))
        order = list(range(len(group)))
        rng.shuffle(order)
        selected = group.iloc[order[: min(len(group), maximum_real)]].copy()
        selected["sample_kind"] = "real"
        selected["source_object_id"] = selected["object_id"]
        selected["synthetic_seed"] = ""
        selected["synthetic_replica_index"] = ""
        selected["augmentation_node"] = node_key
        approved = []
        rejection_reasons = []
        for path in selected["image_path"].astype(str):
            if source_validator is None:
                is_approved, reason = True, ""
            else:
                is_approved, reason = source_validator(path)
            approved.append(bool(is_approved))
            rejection_reasons.append(str(reason or ""))
        selected["augmentation_source_approved"] = approved
        selected["augmentation_rejection_reason"] = rejection_reasons
        parts.append(selected)

        desired = target_per_class - len(selected)
        source_pool = selected.loc[selected["augmentation_source_approved"]].reset_index(drop=True)
        capacity = len(source_pool) * max_synthetic_per_source
        synthetic_count = min(desired, capacity)
        if synthetic_count <= 0:
            continue

        source_order = list(range(len(source_pool)))
        rng.shuffle(source_order)
        synthetic_rows = []
        replica_counts = [0] * len(source_pool)
        for synthetic_index in range(synthetic_count):
            source_position = source_order[synthetic_index % len(source_order)]
            replica_counts[source_position] += 1
            source = source_pool.iloc[source_position].copy()
            replica = replica_counts[source_position]
            recipe_seed = stable_seed(
                seed,
                node_key,
                class_label,
                source["object_id"],
                replica,
            )
            source["sample_kind"] = "synthetic"
            source["source_object_id"] = source["object_id"]
            source["synthetic_seed"] = str(recipe_seed)
            source["synthetic_replica_index"] = str(replica)
            source["augmentation_node"] = node_key
            synthetic_rows.append(source)
        parts.append(pd.DataFrame(synthetic_rows))

    output = pd.concat(parts, ignore_index=True)
    shuffle_rng = np.random.default_rng(stable_seed(seed, node_key, "shuffle"))
    order = shuffle_rng.permutation(len(output))
    return output.iloc[order].reset_index(drop=True)
