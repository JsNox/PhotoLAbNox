from pathlib import Path
import re
import os

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
PROJECTS = ROOT / "projects"
PRESETS = ROOT / "presets"
CACHE = ROOT / ".cache"
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_PIXELS = 24_000_000


def setup():
    for path in (MODELS, OUTPUTS, PROJECTS, PRESETS, CACHE):
        path.mkdir(exist_ok=True, parents=True)
        path.chmod(0o700)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["GRADIO_TEMP_DIR"] = str(CACHE)
    os.environ["U2NET_HOME"] = str(MODELS / "rembg")


def safe_name(name):
    name = re.sub(r"[^\w -]", "", str(name), flags=re.UNICODE).strip()[:64]
    if not name or name in (".", ".."):
        raise ValueError("Escribe un nombre válido para guardar.")
    return name


def confined(root, name, suffix):
    path = root / (safe_name(name) + suffix)
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError("Ruta local no permitida.")
    return path
