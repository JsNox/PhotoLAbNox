from io import BytesIO
import json
import zipfile
import numpy as np
import pytest
from PIL import Image
from app.core.image_processor import EditorSession
from app.core.color_correction import adjust
from app.core.project_store import save_project, load_project, list_projects
from app.core.presets import save_preset, load_preset, apply_preset, BUILTINS
from app.core.batch import process_batch
from app.core.export import encode, save_copy, variants
from app.utils.paths import safe_name, confined


def test_project_roundtrip(photo, tmp_path):
    s = EditorSession().open(photo)
    s.commit(adjust(photo, exposure=0.5), "Exposición")
    save_project(s, "Proyecto / uno", tmp_path)
    assert list_projects(tmp_path) == ["Proyecto  uno"]
    loaded = load_project("Proyecto / uno", tmp_path)
    assert loaded.operations == ["Exposición"]
    assert np.array_equal(loaded.current, s.current)
    loaded.project = None
    loaded.undo()
    assert np.array_equal(loaded.current, photo)
    assert not list(tmp_path.glob(".lpl-*"))


def test_autosave(photo, tmp_path, monkeypatch):
    import app.core.project_store as store

    real_save = store.save_project
    monkeypatch.setattr(store, "save_project", lambda s, n: real_save(s, n, tmp_path))
    s = EditorSession().open(photo)
    s.project = "auto"
    s.commit(adjust(photo, brightness=10), "Brillo")
    assert np.array_equal(load_project("auto", tmp_path).current, s.current)
    s.reset()
    assert np.array_equal(load_project("auto", tmp_path).current, photo)


def test_corrupt_projects(tmp_path):
    (tmp_path / "roto.lpl").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="dañado"):
        load_project("roto", tmp_path)
    with zipfile.ZipFile(tmp_path / "version.lpl", "w") as z:
        z.writestr("manifest.json", json.dumps({"version": 999}))
    with pytest.raises(ValueError, match="incompatible"):
        load_project("version", tmp_path)
    with zipfile.ZipFile(tmp_path / "json.lpl", "w") as z:
        z.writestr("manifest.json", "null")
    with pytest.raises(ValueError):
        load_project("json", tmp_path)


def test_sanitization(tmp_path):
    assert "/" not in safe_name("../../test")
    with pytest.raises(ValueError):
        safe_name("...")
    (tmp_path / "evil.lpl").symlink_to(tmp_path.parent / "elsewhere")
    with pytest.raises(ValueError):
        confined(tmp_path, "evil", ".lpl")


def test_presets(photo, tmp_path, person):
    p = load_preset("Natural")
    save_preset("Mi estilo", p, tmp_path)
    assert load_preset("Mi estilo", tmp_path) == p
    for p in BUILTINS.values():
        assert apply_preset(photo, p).size == photo.size
    shaped = load_preset("Natural")
    shaped["bust"] = 10
    assert apply_preset(photo, shaped, lambda im: person).size == photo.size
    with pytest.raises(ValueError):
        save_preset("Natural", p, tmp_path)
    (tmp_path / "bad.json").write_text("[]")
    with pytest.raises(ValueError):
        load_preset("bad", tmp_path)


def test_batch(photo, tmp_path):
    source = tmp_path / "input.png"
    photo.save(source)
    p = load_preset("Natural")
    path = process_batch([source, source], p, {"text": "LAB"}, tmp_path)
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist()) == 2
        for n in z.namelist():
            assert "/" not in n and n.endswith(".jpg")
            im = Image.open(BytesIO(z.read(n)))
            assert not im.getexif()
    with pytest.raises(ValueError):
        process_batch([source] * 101, p, root=tmp_path)
    with pytest.raises(ValueError):
        process_batch([], p, root=tmp_path)


def test_variants_and_exif(photo, tmp_path):
    exif = Image.Exif()
    exif[315] = "Private author"
    exif[34853] = {1: "N", 2: (1.0, 2.0, 3.0)}
    photo.info["exif"] = exif.tobytes()
    for fmt in ("JPEG", "PNG", "WEBP"):
        clean = Image.open(BytesIO(encode(photo, fmt, True)))
        assert not clean.getexif()
        retained = Image.open(BytesIO(encode(photo, fmt, False)))
        assert retained.getexif().get(315) == "Private author"
    paths = variants(photo, "LAB", 16, 90, root=tmp_path)
    assert len(set(paths)) == 3 and all(p.is_file() for p in paths)
    assert save_copy(photo, root=tmp_path) != save_copy(photo, root=tmp_path)
