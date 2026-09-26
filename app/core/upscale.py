"""RealESRGAN x2 via Spandrel, tiled in memory with feathered overlap."""

import threading
import numpy as np
from PIL import Image
from app.utils.image_utils import rgb
from app.models.model_manager import require_model, UPSCALE_SHA256
from app.utils.device import choose_device

_LOCK = threading.Lock()
_CACHE = {}


def tiled_x2(im, predict, tile=256, overlap=24):
    if tile not in (128, 256, 384, 512):
        raise ValueError("El tile debe ser 128, 256, 384 o 512.")
    if im.width * im.height * 4 > 64_000_000:
        raise ValueError("Upscale 2× superaría el límite de 64 megapíxeles.")
    a = rgb(im).astype(np.float32) / 255
    h, w = a.shape[:2]
    out = np.zeros((h * 2, w * 2, 3), np.float32)
    weights = np.zeros((h * 2, w * 2, 1), np.float32)
    step = tile - 2 * overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = a[y : min(y + tile, h), x : min(x + tile, w)]
            ph, pw = patch.shape[:2]
            # RRDB pixel-unshuffle requires even input dimensions; also guard tiny tiles.
            dh = max(2, ph + (ph % 2)) - ph
            dw = max(2, pw + (pw % 2)) - pw
            padded = np.pad(patch, ((0, dh), (0, dw), (0, 0)), mode="edge")
            enhanced = np.asarray(predict(padded), np.float32)[: ph * 2, : pw * 2]
            if enhanced.shape != (ph * 2, pw * 2, 3):
                raise ValueError("El modelo no produjo una salida RGB 2× válida.")
            wy = np.ones(ph * 2, np.float32)
            wx = np.ones(pw * 2, np.float32)
            fade = min(overlap * 2, len(wy) // 2, len(wx) // 2)
            if fade:
                ramp = np.linspace(0.01, 1, fade, dtype=np.float32)
                if y > 0:
                    wy[:fade] = ramp
                if y + ph < h:
                    wy[-fade:] = ramp[::-1]
                if x > 0:
                    wx[:fade] = ramp
                if x + pw < w:
                    wx[-fade:] = ramp[::-1]
            weight = (wy[:, None] * wx[None, :])[..., None]
            out[y * 2 : (y + ph) * 2, x * 2 : (x + pw) * 2] += enhanced * weight
            weights[y * 2 : (y + ph) * 2, x * 2 : (x + pw) * 2] += weight
    out /= np.maximum(weights, 1e-8)
    result = Image.fromarray(np.uint8(np.clip(out * 255, 0, 255)))
    if im.mode == "RGBA":
        result.putalpha(
            im.getchannel("A").resize(result.size, Image.Resampling.LANCZOS)
        )
    result.info.update(im.info)
    return result


def upscale(im, tile=256):
    if im.width * im.height * 4 > 64_000_000:
        raise ValueError("Upscale 2× superaría el límite de 64 megapíxeles.")
    try:
        import torch
        from spandrel import ModelLoader
    except ImportError as exc:
        raise ValueError(
            "Instala requirements-upscale.txt para activar RealESRGAN."
        ) from exc
    with _LOCK:
        if not _CACHE:
            path = require_model("upscale/RealESRGAN_x2plus.pth", UPSCALE_SHA256)
            model = ModelLoader().load_from_file(str(path))
            if model.scale != 2:
                raise ValueError("Se requiere exactamente el modelo RealESRGAN 2×.")
            _CACHE.update(model=model, device=choose_device(torch))
        model, device = _CACHE["model"], _CACHE["device"]

        def run(target):
            dtype = torch.float16 if target == "cuda" else torch.float32
            model.to(device=target, dtype=dtype).eval()

            def predict(patch):
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))
                    .unsqueeze(0)
                    .to(device=target, dtype=dtype)
                )
                with torch.inference_mode():
                    output = model(tensor)
                return output.squeeze(0).float().cpu().numpy().transpose(1, 2, 0)

            return tiled_x2(im, predict, int(tile))

        try:
            return run(device), device
        except RuntimeError:
            if device != "mps":
                raise
            _CACHE["device"] = "cpu"
            model.to(device="cpu", dtype=torch.float32)
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
            return run("cpu"), "cpu (fallback desde MPS)"
