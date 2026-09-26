from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image, ImageDraw
from app.utils.image_utils import mask_array
from app.core.background import clean_plate

NAMES = ("Hombros", "Busto", "Cintura", "Cadera", "Límite inferior")


@dataclass
class Region:
    name: str
    y: int
    center: float
    width: float
    left: int
    right: int


def row_region(mask, y, name, x=None):
    a = np.asarray(mask)
    if a.max(initial=0) <= 1:
        a = a * 255
    y = int(np.clip(y, 0, a.shape[0] - 1))
    indices = np.flatnonzero(a[y] > 100)
    if not len(indices):
        raise ValueError(
            "La fila seleccionada no cruza la silueta. Prueba otra altura."
        )
    # Keep the contiguous silhouette run nearest the chosen torso center.
    runs = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    x = a.shape[1] / 2 if x is None else x
    run = min(runs, key=lambda r: abs((r[0] + r[-1]) / 2 - x))
    left, right = int(run[0]), int(run[-1])
    return Region(name, y, (left + right) / 2, max(2, right - left + 1), left, right)


def estimate(mask):
    a = np.asarray(mask)
    ys, xs = np.where(a > (100 if a.max(initial=0) > 1 else 0.4))
    if not len(ys):
        raise ValueError("No se detectó una persona en la máscara.")
    top, bottom = int(ys.min()), int(ys.max())
    regions = []
    for name, fraction in zip(NAMES, (0.24, 0.36, 0.51, 0.65, 0.79)):
        desired = top + round((bottom - top) * fraction)
        y = int(ys[np.argmin(abs(ys - desired))])
        regions.append(row_region(mask, y, name, float(np.median(xs))))
    if len({r.y for r in regions}) != 5:
        raise ValueError("Silueta demasiado pequeña para estimar cinco regiones.")
    return regions


def calibrate(mask, points):
    if len(points) != 5:
        raise ValueError(
            "Marca cinco puntos: hombros, busto, cintura, cadera y límite inferior."
        )
    if any(points[i][1] >= points[i + 1][1] for i in range(4)):
        raise ValueError("Los cinco clics deben ir de arriba hacia abajo.")
    return [row_region(mask, y, name, x) for name, (x, y) in zip(NAMES, points)]


def overlay(im, regions):
    out = im.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for r in regions:
        draw.line(
            (r.left, r.y, r.right, r.y), fill="#67e8c3", width=max(2, im.width // 500)
        )
        draw.ellipse((r.center - 4, r.y - 4, r.center + 4, r.y + 4), fill="#ffbd69")
        draw.text(
            (r.left, max(0, r.y - 13)),
            r.name,
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
    return out


def effective(bust=0, waist=0, hip=0, curves=0, intensity=60):
    factor = np.clip(intensity, 0, 100) / 100
    return np.clip(
        np.array([bust, waist - curves * 0.35, hip + curves * 0.35]) * factor / 100,
        -0.18,
        0.18,
    )


def fields(size, regions, amounts):
    w, h = size
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    rows = np.array([r.y for r in regions])
    center = np.interp(np.arange(h), rows, [r.center for r in regions])[:, None]
    width = np.interp(np.arange(h), rows, [r.width for r in regions])[:, None]
    # Piecewise interpolation follows the torso's inclined axis and independent widths.
    gains = np.interp(np.arange(h), rows, [0, *amounts, 0], left=0, right=0)[:, None]
    t = np.clip(abs(xx - center) / (width * 0.85), 0, 1)
    horizontal = 1 - t * t * (3 - 2 * t)
    delta = (xx - center) * (1 / (1 + gains) - 1) * horizontal
    return (xx + delta).astype(np.float32), yy


def composite_warp(im, mask, mapx, mapy, plate=None):
    m = mask_array(mask, im.size)
    if plate is None:
        plate = clean_plate(im, mask)
    src = np.array(im.convert("RGBA"), dtype=np.float32)
    original_alpha = src[..., 3] / 255
    alpha = m * original_alpha
    premul = src[..., :3] * alpha[..., None]
    warped = cv2.remap(
        premul, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    wa = cv2.remap(alpha, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    base = np.array(plate.convert("RGB"), dtype=np.float32)
    comp = warped + base * (1 - wa[..., None])
    # Outside the union of old/new person, background pixels remain exactly unchanged.
    affected = (m > 0) | (wa > 0)
    orig = np.array(im.convert("RGB"))
    comp[~affected] = orig[~affected]
    out = Image.fromarray(np.uint8(np.clip(comp, 0, 255)))
    if im.mode == "RGBA":
        bg_alpha = original_alpha * (1 - m)
        out.putalpha(
            Image.fromarray(np.uint8(np.clip(wa + bg_alpha * (1 - wa), 0, 1) * 255))
        )
    out.info.update(im.info)
    return out, np.uint8(
        np.clip(cv2.remap(m, mapx, mapy, cv2.INTER_LINEAR), 0, 1) * 255
    )


def reshape(im, mask, regions=None, bust=0, waist=0, hip=0, curves=0, intensity=60):
    regions = regions or estimate(mask)
    amounts = effective(bust, waist, hip, curves, intensity)
    if not np.any(amounts):
        return im.copy(), np.uint8(mask_array(mask, im.size) * 255), amounts
    mx, my = fields(im.size, regions, amounts)
    out, new_mask = composite_warp(im, mask, mx, my)
    return out, new_mask, amounts
