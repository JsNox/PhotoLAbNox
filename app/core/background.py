from PIL import Image, ImageFilter, ImageOps, ImageColor
import numpy as np
from app.utils.image_utils import mask_array
from app.core.segmentation import combine
from app.core.inpainting import heal


def clean_plate(im, mask):
    # Bounded working resolution: full-person Telea is expensive at camera resolution.
    scale = min(1, 768 / max(im.size))
    size = tuple(max(1, round(v * scale)) for v in im.size)
    small = im.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    m = Image.fromarray(np.uint8(mask_array(mask, im.size) * 255)).resize(size)
    return heal(small, m, radius=5, expansion=3).resize(
        im.size, Image.Resampling.BICUBIC
    )


def replace(
    im,
    person,
    mode="Transparente",
    color="#eeeeee",
    photo=None,
    blur=20,
    feather=2,
    keep=None,
):
    mask = combine(person, keep, im.size)
    m = mask_array(mask, im.size, feather)
    # Input alpha must also constrain the foreground.
    if im.mode == "RGBA":
        m *= np.array(im.getchannel("A")) / 255
    alpha = Image.fromarray(np.uint8(m * 255))
    if mode == "Transparente":
        out = im.convert("RGBA")
        out.putalpha(alpha)
    else:
        if mode == "Color sólido":
            base = Image.new("RGB", im.size, ImageColor.getrgb(color))
        elif mode == "Fotografía":
            if photo is None:
                raise ValueError("Carga la fotografía de fondo.")
            base = ImageOps.fit(photo.convert("RGB"), im.size)
        else:
            base = clean_plate(im, mask).filter(ImageFilter.GaussianBlur(blur))
        out = Image.composite(im.convert("RGB"), base, alpha)
    out.info.update(im.info)
    return out
