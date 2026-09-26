"""Local-only model discovery. Never invokes a downloader."""

import hashlib
from functools import lru_cache
from app.utils.paths import MODELS

UPSCALE_SHA256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"


def require_model(relative, checksum=None):
    path = MODELS / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"Falta el modelo local models/{relative}. Consulta la instalación en README.md."
        )
    if checksum:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError(
                "SHA-256 incorrecto: el peso de upscale no es el esperado."
            )
    return path


@lru_cache(maxsize=1)
def segmentation_session():
    path = require_model("rembg/u2net_human_seg.onnx")
    try:
        # Use rembg's preprocessing/postprocessing but bypass its download_models hook.
        import onnxruntime as ort

        ort.disable_telemetry_events()
        from rembg.sessions.u2net_human_seg import U2netHumanSegSession

        session = U2netHumanSegSession.__new__(U2netHumanSegSession)
        session.model_name = "u2net_human_seg"
        session.inner_session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        return session
    except ImportError as exc:
        raise ValueError(
            "Instala requirements-background.txt para activar la segmentación."
        ) from exc
