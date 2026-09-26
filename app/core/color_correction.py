import cv2
import numpy as np
from app.utils.image_utils import rgb, result_like

DEFAULTS = dict(
    exposure=0.0,
    brightness=0.0,
    contrast=0.0,
    saturation=0.0,
    temperature=0.0,
    shadows=0.0,
    highlights=0.0,
    sharpness=0.0,
    denoise=0.0,
)


def adjust(im, **values):
    p = DEFAULTS | values
    if not any(p.values()):
        return im.copy()
    a = rgb(im).astype(np.float32) / 255
    a *= 2 ** np.clip(p["exposure"], -3, 3)
    a += p["brightness"] / 200
    a = (a - 0.5) * (1 + p["contrast"] / 100) + 0.5
    a[..., 0] += p["temperature"] / 500
    a[..., 2] -= p["temperature"] / 500
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    a += (
        p["shadows"] / 250 * (1 - np.clip(lum, 0, 1)) ** 2
        + p["highlights"] / 250 * np.clip(lum, 0, 1) ** 2
    )[..., None]
    gray = np.sum(a * np.array([0.2126, 0.7152, 0.0722]), axis=2, keepdims=True)
    a = gray + (a - gray) * (1 + p["saturation"] / 100)
    a = np.uint8(np.clip(a * 255, 0, 255))
    if p["denoise"] > 0:
        a = cv2.bilateralFilter(a, 5, float(p["denoise"]) * 0.6, 5)
    if p["sharpness"] > 0:
        blur = cv2.GaussianBlur(a, (0, 0), 1)
        a = cv2.addWeighted(a, 1 + p["sharpness"] / 100, blur, -p["sharpness"] / 100, 0)
    return result_like(im, a)


def intelligent(im):
    a = rgb(im)
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    lab[..., 0] = cv2.addWeighted(lab[..., 0], 0.65, clahe.apply(lab[..., 0]), 0.35, 0)
    out = result_like(im, cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    return adjust(out, saturation=4, sharpness=12, shadows=4, highlights=-3)
