from uuid import uuid4
import zipfile
from app.utils.paths import OUTPUTS
from app.utils.image_utils import load_image
from app.core.presets import apply_preset
from app.core.local_adjustments import watermark
from app.core.export import encode


def process_batch(files, preset, mark=None, root=OUTPUTS, segmenter=None):
    if not files or len(files) > 100:
        raise ValueError("Selecciona entre 1 y 100 fotografías.")
    root.mkdir(exist_ok=True, parents=True)
    path = root / (uuid4().hex + ".zip")
    try:
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
            for index, source in enumerate(files):
                im = apply_preset(load_image(source), preset, segmenter=segmenter)
                if mark:
                    im = watermark(im, **mark)
                archive.writestr(
                    f"foto_{index + 1:03d}_{uuid4().hex[:8]}.jpg",
                    encode(im, "JPEG", True),
                )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path
