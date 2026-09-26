import subprocess
import sys
import numpy as np
import pytest
from PIL import Image
from app.core.upscale import tiled_x2
from app.models.model_manager import require_model
from app.ui.gradio_app import create_app, uploaded, color_action, reshape_action, render
from app.core.image_processor import EditorSession


@pytest.mark.parametrize("size", [(1, 1), (65, 99), (301, 271), (600, 399)])
@pytest.mark.parametrize("tile", [128, 256, 384, 512])
def test_upscale_tiles(size, tile):
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (size[1], size[0], 4), dtype=np.uint8)
    im = Image.fromarray(a)

    def predict(p):
        return np.repeat(np.repeat(p, 2, 0), 2, 1)

    out = tiled_x2(im, predict, tile)
    assert out.size == (size[0] * 2, size[1] * 2)
    expected = np.repeat(np.repeat(a[:, :, :3], 2, 0), 2, 1)
    assert np.abs(np.array(out)[:, :, :3].astype(int) - expected.astype(int)).max() <= 1
    assert out.mode == "RGBA"


def test_upscale_limit_and_missing():
    with pytest.raises(ValueError, match="64 mega"):
        tiled_x2(Image.new("RGB", (4001, 4000)), None)
    with pytest.raises(ValueError, match="Falta el modelo"):
        require_model("missing.pth")


def test_checksum(tmp_path, monkeypatch):
    import app.models.model_manager as m

    monkeypatch.setattr(m, "MODELS", tmp_path)
    (tmp_path / "bad.pth").write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA-256"):
        require_model("bad.pth", "0" * 64)


def test_network_block():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.utils.privacy import disable_outbound_network; import socket; disable_outbound_network(); socket.getaddrinfo('example.com',443)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "bloquea DNS externo" in result.stderr


def test_upload_arbitrary_path_rejected():
    with pytest.raises(ValueError):
        uploaded("/etc/passwd")


def test_ui_build():
    demo = create_app()
    assert demo.analytics_enabled is False
    assert len(demo.blocks) > 100
    assert not any("fonts.googleapis.com" in str(v) for v in demo.config.values())
    demo.close()


def test_controller_flow(photo, person):
    s = EditorSession().open(photo)
    assert "pequeño" in color_action(s, *([0] * 9))
    s.mask = person
    from app.core.body_reshape import estimate

    s.regions = estimate(person)
    message = reshape_action(s, 20, -20, 10, 5, 60)
    assert "busto" in message and len(s.operations) == 2
    assert len(render(s)) == 8


def test_mps_fallback_to_cpu(photo, monkeypatch):
    import sys
    import types
    from contextlib import nullcontext
    import app.core.upscale as module

    calls = []

    class Tensor:
        def __init__(self, a):
            self.a = a

        def unsqueeze(self, axis):
            return Tensor(np.expand_dims(self.a, axis))

        def squeeze(self, axis):
            return Tensor(np.squeeze(self.a, axis))

        def to(self, **kw):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class Model:
        def to(self, device, dtype):
            calls.append((device, dtype))
            if device == "mps":
                raise RuntimeError("simulated unsupported MPS operation")
            return self

        def eval(self):
            return self

        def __call__(self, t):
            return Tensor(np.repeat(np.repeat(t.a, 2, 2), 2, 3))

    fake = types.SimpleNamespace(
        float32="fp32",
        float16="fp16",
        from_numpy=Tensor,
        inference_mode=nullcontext,
        mps=types.SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(
        sys.modules, "spandrel", types.SimpleNamespace(ModelLoader=object)
    )
    monkeypatch.setattr(module, "_CACHE", {"model": Model(), "device": "mps"})
    out, device = module.upscale(photo, 128)
    assert device == "cpu (fallback desde MPS)"
    assert out.size == (photo.width * 2, photo.height * 2)
    assert module._CACHE["device"] == "cpu"
    assert calls[0] == ("mps", "fp32") and calls[-1] == ("cpu", "fp32")


def test_device_selection():
    from types import SimpleNamespace as NS
    from app.utils.device import choose_device

    assert choose_device(NS(cuda=NS(is_available=lambda: True))) == "cuda"
    assert (
        choose_device(
            NS(
                cuda=NS(is_available=lambda: False),
                backends=NS(mps=NS(is_available=lambda: True)),
            )
        )
        == "mps"
    )
    assert (
        choose_device(NS(cuda=NS(is_available=lambda: False), backends=NS())) == "cpu"
    )


def test_gradio_template_has_no_remote_resources():
    import re
    from app.utils.privacy import localize_gradio_template
    from gradio.routes import templates

    localize_gradio_template()
    source, _, _ = templates.env.loader.get_source(templates.env, "frontend/index.html")
    assert not re.search(r"<script\b[^>]*src=[\"\']https?://", source, re.I)
    assert not re.search(r"<link\b[^>]*https?://", source, re.I)


def test_security_headers():
    import asyncio
    from app.utils.privacy import LocalSecurityHeaders

    messages = []

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        messages.append(message)

    asyncio.run(LocalSecurityHeaders(inner)({"type": "http"}, None, send))
    headers = dict(messages[0]["headers"])
    assert b"connect-src 'self'" in headers[b"content-security-policy"]
    assert headers[b"referrer-policy"] == b"no-referrer"
