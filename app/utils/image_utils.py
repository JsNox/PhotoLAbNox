from io import BytesIO
from pathlib import Path
import warnings
import numpy as np
import cv2
from PIL import Image, ImageOps
from app.utils.paths import MAX_FILE_BYTES, MAX_PIXELS

Image.MAX_IMAGE_PIXELS = 64_000_000


def load_image(source, max_pixels=MAX_PIXELS):
    if isinstance(source, Image.Image):
        im = source.copy()
    else:
        if (
            isinstance(source, (str, Path))
            and Path(source).stat().st_size > MAX_FILE_BYTES
        ):
            raise ValueError("Archivo demasiado grande: máximo 40 MB.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            try:
                with Image.open(source) as opened:
                    if opened.format not in ("JPEG", "PNG", "WEBP"):
                        raise ValueError("Solo se aceptan JPG, PNG y WEBP.")
                    if opened.width * opened.height > max_pixels:
                        raise ValueError(
                            f"La imagen supera el límite de {max_pixels // 1_000_000} megapíxeles."
                        )
                    fmt = opened.format
                    im = ImageOps.exif_transpose(opened)
                    im.load()
                    im.info["source_format"] = fmt
            except (
                OSError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ) as exc:
                raise ValueError(
                    "No se puede leer la imagen o supera el límite seguro."
                ) from exc
    if im.width * im.height > max_pixels:
        raise ValueError("La imagen supera el límite de resolución.")
    info = im.info.copy()
    im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
    # Convert tagged images to sRGB before using RGB/LAB operations.
    if info.get("icc_profile"):
        try:
            from PIL import ImageCms

            alpha = im.getchannel("A") if im.mode == "RGBA" else None
            srgb = ImageCms.createProfile("sRGB")
            im = ImageCms.profileToProfile(
                im.convert("RGB"),
                ImageCms.ImageCmsProfile(BytesIO(info["icc_profile"])),
                srgb,
                outputMode="RGB",
            )
            if alpha is not None:
                im.putalpha(alpha)
            info["icc_profile"] = ImageCms.ImageCmsProfile(srgb).tobytes()
        except (OSError, ValueError):
            info.pop("icc_profile", None)
    im.info.update(info)
    return im


def rgb(im):
    return np.array(im.convert("RGB"))


def result_like(im, pixels):
    out = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    if im.mode == "RGBA":
        out.putalpha(im.getchannel("A"))
    out.info.update(im.info)
    return out


def mask_array(mask, size, feather=0):
    if mask is None:
        raise ValueError("Primero pinta una máscara o segmenta la persona.")
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L").resize(size, Image.Resampling.BILINEAR))
    arr = np.asarray(mask, dtype=np.float32)
    if arr.shape != (size[1], size[0]):
        raise ValueError(
            "La máscara no coincide con la imagen actual; vuelve a crearla."
        )
    if arr.max(initial=0) > 1:
        arr /= 255
    if feather > 0:
        arr = cv2.GaussianBlur(arr, (0, 0), max(0.1, float(feather)))
    return np.clip(arr, 0, 1)


def blend(im, edited, mask, feather=0):
    m = mask_array(mask, im.size, feather)[..., None]
    return result_like(im, rgb(im) * (1 - m) + rgb(edited) * m)


def editor_mask(value, size):
    if not value or not value.get("layers"):
        raise ValueError("Pinta sobre la fotografía antes de aplicar la herramienta.")
    mask = np.zeros((size[1], size[0]), np.float32)
    for layer in value["layers"]:
        layer = layer if isinstance(layer, Image.Image) else Image.fromarray(layer)
        if layer.size != size:
            raise ValueError(
                "Actualiza el lienzo: su tamaño no coincide con el resultado."
            )
        if layer.mode != "RGBA":
            raise ValueError("La capa pintada debe tener transparencia.")
        mask = np.maximum(mask, np.array(layer.getchannel("A")) / 255.0)
    if not mask.any():
        raise ValueError("La máscara está vacía.")
    return mask


def png_bytes(im):
    buf = BytesIO()
    # Projects retain ICC but metadata is stored explicitly, never pickle.
    im.save(buf, "PNG", icc_profile=im.info.get("icc_profile"))
    return buf.getvalue()
