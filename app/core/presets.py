import json
import os
import tempfile
from app.utils.paths import PRESETS, confined
from app.core.color_correction import DEFAULTS, adjust
from app.core.skin_retouch import retouch

BASE = dict(
    color=DEFAULTS,
    retouch=0,
    texture=70,
    bust=0,
    waist=0,
    hip=0,
    curves=0,
    intensity=60,
)
BUILTINS = {
    "Natural": BASE
    | {"color": DEFAULTS | dict(shadows=6, sharpness=10), "retouch": 20},
    "Cálido": BASE
    | {"color": DEFAULTS | dict(temperature=16, saturation=5), "retouch": 25},
    "Nocturno": BASE
    | {
        "color": DEFAULTS | dict(exposure=0.25, shadows=16, temperature=-8, denoise=25),
        "retouch": 15,
    },
    "Editorial": BASE
    | {
        "color": DEFAULTS | dict(contrast=12, saturation=-8, sharpness=18),
        "retouch": 30,
    },
}


def validate(p):
    if (
        not isinstance(p, dict)
        or set(p) != set(BASE)
        or not isinstance(p["color"], dict)
        or set(p["color"]) != set(DEFAULTS)
    ):
        raise ValueError("Preset incompleto o incompatible.")
    for key, value in p["color"].items():
        limit = 3 if key == "exposure" else 100
        if not isinstance(value, (float, int)) or not -limit <= value <= limit:
            raise ValueError("Ajuste de color fuera del rango permitido.")
    for key in set(BASE) - {"color"}:
        low, high = (
            (-20, 20)
            if key in ("bust", "waist", "hip")
            else (0, 20)
            if key == "curves"
            else (0, 100)
        )
        if not isinstance(p[key], (int, float)) or not low <= p[key] <= high:
            raise ValueError("Valor de preset fuera del rango permitido.")
    return p


def save_preset(name, values, root=PRESETS):
    values = validate(values)
    if name in BUILTINS:
        raise ValueError("Usa otro nombre para conservar los presets incluidos.")
    root.mkdir(exist_ok=True, parents=True)
    path = confined(root, name, ".json")
    fd, tmp = tempfile.mkstemp(dir=root, prefix=".preset-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(values, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_preset(name, root=PRESETS):
    if name in BUILTINS:
        return json.loads(json.dumps(BUILTINS[name]))
    try:
        path = confined(root, name, ".json")
        if path.stat().st_size > 32_000:
            raise ValueError("Preset demasiado grande.")
        return validate(json.loads(path.read_text()))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("Preset dañado o incompatible.") from exc


def list_presets(root=PRESETS):
    return list(BUILTINS) + sorted(
        p.stem
        for p in root.glob("*.json")
        if not p.is_symlink() and p.stem not in BUILTINS
    )


def apply_preset(im, p, segmenter=None):
    p = validate(p)
    out = retouch(adjust(im, **p["color"]), p["retouch"], p["texture"])
    if any(p[k] for k in ("bust", "waist", "hip", "curves")):
        from app.core.segmentation import segment
        from app.core.body_reshape import reshape

        mask = (segmenter or segment)(out)
        out, _, _ = reshape(
            out,
            mask,
            **{k: p[k] for k in ("bust", "waist", "hip", "curves", "intensity")},
        )
    return out
