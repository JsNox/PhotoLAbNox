import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from app.utils.image_utils import rgb, result_like, blend, mask_array


def dodge_burn(im, mask, amount=20, feather=8):
    if amount == 0:
        return im.copy()
    lab = cv2.cvtColor(rgb(im).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    m = mask_array(mask, im.size, feather)
    # Smooth gain on L only: chrominance and fine luminance structure retained.
    lab[..., 0] = np.clip(
        lab[..., 0] * 2 ** (m * np.clip(amount, -100, 100) / 100), 0, 100
    )
    return result_like(im, cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255)


def censor(im, mask, mode="Desenfoque", intensity=20, feather=3):
    if intensity <= 0:
        return im.copy()
    if mode == "Pixelado":
        block = max(2, int(intensity))
        small = (max(1, im.width // block), max(1, im.height // block))
        target = im.resize(small, Image.Resampling.BOX).resize(
            im.size, Image.Resampling.NEAREST
        )
    else:
        target = im.filter(ImageFilter.GaussianBlur(intensity))
    return blend(im, target, mask, feather)


def watermark(
    im, text="Local Photo Lab", size=32, opacity=40, position="Inferior derecha"
):
    if not text.strip() or opacity <= 0:
        return im.copy()
    layer = Image.new("RGBA", im.size)
    draw = ImageDraw.Draw(layer)
    font = ImageFont.load_default(size=max(8, int(size)))
    text = text[:180]
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    margin = max(5, int(min(im.size) * 0.025))
    x = margin if "izquierda" in position else max(margin, im.width - w - margin)
    y = margin if "Superior" in position else max(margin, im.height - h - margin)
    if position == "Centro":
        x, y = (im.width - w) // 2, (im.height - h) // 2
    draw.text(
        (x, y - box[1]),
        text,
        font=font,
        fill=(255, 255, 255, int(255 * opacity / 100)),
        stroke_width=1,
        stroke_fill=(0, 0, 0, int(150 * opacity / 100)),
    )
    out = Image.alpha_composite(im.convert("RGBA"), layer)
    if im.mode == "RGB":
        out = out.convert("RGB")
    out.info.update(im.info)
    return out
