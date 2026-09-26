import cv2
import numpy as np
from app.utils.image_utils import rgb, result_like

HUES = {
    "Rojo": 0,
    "Naranja": 30,
    "Amarillo": 60,
    "Verde": 120,
    "Cian": 180,
    "Azul": 240,
    "Morado": 275,
    "Magenta": 310,
}


def hsl(im, channel="Todos", hue=0, saturation=0, lightness=0):
    if hue == saturation == lightness == 0:
        return im.copy()
    a = cv2.cvtColor(rgb(im).astype(np.float32) / 255, cv2.COLOR_RGB2HLS)
    if channel == "Todos":
        m = np.ones(a.shape[:2])
    else:
        d = np.abs((a[..., 0] - HUES[channel] + 180) % 360 - 180)
        m = np.clip(1 - d / 40, 0, 1)
        m = m * m * (3 - 2 * m)
    a[..., 0] = (a[..., 0] + hue * m) % 360
    a[..., 1] = np.clip(a[..., 1] + lightness / 200 * m, 0, 1)
    a[..., 2] = np.clip(a[..., 2] * (1 + saturation / 100 * m), 0, 1)
    return result_like(im, cv2.cvtColor(a, cv2.COLOR_HLS2RGB) * 255)


def white_balance(im, point):
    x, y = map(int, point)
    if not (0 <= x < im.width and 0 <= y < im.height):
        raise ValueError("Selecciona un punto dentro de la fotografía.")
    a = rgb(im).astype(np.float32)
    sample = a[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4].mean(axis=(0, 1))
    if sample.min() < 8:
        raise ValueError(
            "La muestra es demasiado oscura; elige una zona gris o blanca."
        )
    gain = np.clip(sample.mean() / sample, 0.5, 2)
    return result_like(im, a * gain)
