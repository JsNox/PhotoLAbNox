import numpy as np
import pytest
from PIL import Image
from app.core.inpainting import heal
from app.core.image_processor import EditorSession
from app.ui.gradio_app import remove_objects_action, render


def fixture_object():
    pixels = np.full((80, 100, 3), 160, dtype=np.uint8)
    pixels[30:40, 20:30] = (255, 0, 0)
    layer = np.zeros((80, 100, 4), dtype=np.uint8)
    layer[30:40, 20:30] = 255
    return Image.fromarray(pixels), {"layers": [Image.fromarray(layer)]}


def test_object_removed_and_undo():
    im, strokes = fixture_object()
    s = EditorSession().open(im)
    remove_objects_action(s, strokes, 3, 2, False)
    assert np.abs(np.array(s.current)[30:40, 20:30].astype(float) - 160).max() < 8
    assert s.operations == ["Eliminar objetos del fondo"]
    assert np.array_equal(s.original, im)
    assert np.array_equal(np.array(s.current)[:20], np.array(im)[:20])
    assert render(s)[-1]["layers"] == []
    s.undo()
    assert np.array_equal(s.current, im)


def test_protection_applies_after_expansion():
    im, strokes = fixture_object()
    mask = np.array(strokes["layers"][0].getchannel("A"))
    protected = np.zeros_like(mask)
    protected[:, 28:] = 255
    out = heal(im, mask, 3, 8, protected=protected)
    assert np.array_equal(np.array(out)[:, 28:], np.array(im)[:, 28:])
    assert not np.array_equal(out, im)


def test_selection_errors_leave_history_unchanged():
    im, strokes = fixture_object()
    s = EditorSession().open(im)
    with pytest.raises(ValueError, match="Segmenta"):
        remove_objects_action(s, strokes, 3, 2, True)
    s.mask = np.full((80, 100), 255, np.uint8)
    with pytest.raises(ValueError, match="protegida"):
        remove_objects_action(s, strokes, 3, 2, True)
    with pytest.raises(ValueError, match="Pinta"):
        remove_objects_action(s, {"layers": []}, 3, 2, False)
    assert s.operations == []
    assert np.array_equal(s.current, im)


def test_object_removal_autosaves(tmp_path, monkeypatch):
    import app.core.project_store as store

    save = store.save_project
    monkeypatch.setattr(store, "save_project", lambda s, n: save(s, n, tmp_path))
    im, strokes = fixture_object()
    s = EditorSession().open(im)
    s.project = "eliminar"
    remove_objects_action(s, strokes, 3, 2, False)
    loaded = store.load_project("eliminar", tmp_path)
    assert np.array_equal(loaded.current, s.current)
    assert loaded.operations == ["Eliminar objetos del fondo"]
