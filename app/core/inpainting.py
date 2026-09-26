import cv2
import numpy as np
from app.utils.image_utils import rgb, result_like, mask_array, blend


def heal(im, mask, radius=3, expansion=3, protected=None):
    m = np.uint8(mask_array(mask, im.size) > 0.05) * 255
    if expansion:
        k = 2 * int(expansion) + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    if protected is not None:
        m[mask_array(protected, im.size) > 0.05] = 0
    if not np.any(m):
        raise ValueError(
            "La selección está vacía o queda dentro de la persona protegida."
        )
    if np.all(m):
        raise ValueError(
            "Telea necesita una zona sin pintar alrededor de la selección."
        )
    return result_like(
        im, cv2.inpaint(rgb(im), m, max(1, float(radius)), cv2.INPAINT_TELEA)
    )


def clone(im, source, destination, radius=30, opacity=80):
    x, y = map(int, destination)
    sx, sy = map(int, source)
    yy, xx = np.mgrid[: im.height, : im.width].astype(np.float32)
    mapped = cv2.remap(
        rgb(im),
        xx + sx - x,
        yy + sy - y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = np.clip(1 - np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / max(1, radius), 0, 1)
    return blend(im, result_like(im, mapped), mask * opacity / 100)
