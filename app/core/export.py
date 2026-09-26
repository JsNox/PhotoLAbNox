from io import BytesIO
from uuid import uuid4
from PIL import Image, ImageFilter
from app.core.local_adjustments import watermark
from app.utils.paths import OUTPUTS


def encode(im, format="PNG", strip_metadata=True):
    fmt = format.upper()
    if fmt not in ("PNG", "JPEG", "WEBP"):
        raise ValueError("Formato de exportación no permitido.")
    out = im.copy()
    icc = out.info.get("icc_profile")
    exif = out.info.get("exif")
    out.info.clear()
    if fmt == "JPEG":
        if out.mode == "RGBA":
            base = Image.new("RGB", out.size, "white")
            base.paste(out, mask=out.getchannel("A"))
            out = base
        else:
            out = out.convert("RGB")
    args = {"icc_profile": icc} if icc else {}
    if not strip_metadata and exif:
        args["exif"] = exif
    buf = BytesIO()
    out.save(buf, fmt, quality=92, **args)
    return buf.getvalue()


def save_copy(im, format="PNG", strip_metadata=True, root=OUTPUTS):
    root.mkdir(exist_ok=True, parents=True)
    suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[format]
    path = root / (uuid4().hex + suffix)
    with path.open("xb") as stream:
        stream.write(encode(im, format, strip_metadata))
    return path


def variants(
    im,
    text,
    size=32,
    opacity=40,
    position="Inferior derecha",
    format="JPEG",
    strip_metadata=True,
    root=OUTPUTS,
):
    teaser = im.filter(ImageFilter.GaussianBlur(max(25, min(im.size) * 0.065)))
    return [
        save_copy(x, format, strip_metadata, root)
        for x in (im, watermark(im, text, size, opacity, position), teaser)
    ]
