import numpy as np
import pytest
from PIL import Image
from app.core.color_correction import adjust, intelligent
from app.core.skin_retouch import skin_mask, retouch
from app.core.local_adjustments import dodge_burn, censor, watermark
from app.core.advanced_adjustments import hsl, white_balance, HUES
from app.core.inpainting import heal, clone
from app.core.transforms import transform, RATIOS
from app.core.segmentation import segment, combine
from app.core.background import replace
from app.core.image_processor import EditorSession
from app.utils.image_utils import load_image, editor_mask


def test_neutral(photo):
    for result in (adjust(photo), retouch(photo, 0), hsl(photo), transform(photo)):
        assert np.array_equal(result, photo)


def test_history_undo_reset(photo):
    s = EditorSession().open(photo)
    for i in range(30):
        s.commit(adjust(s.current, brightness=1), str(i))
    assert len(s.history) == 25
    assert len(s.operations) == 30
    previous = np.asarray(s.current).copy()
    s.reset()
    assert np.array_equal(s.current, photo)
    s.undo()
    assert np.array_equal(s.current, previous)
    for _ in range(24):
        s.undo()
    with pytest.raises(ValueError):
        s.undo()
    assert np.array_equal(s.original, photo)


def test_import_and_orientation(tmp_path):
    im = Image.new("RGB", (20, 10), "red")
    exif = Image.Exif()
    exif[274] = 6
    p = tmp_path / "x.jpg"
    im.save(p, exif=exif)
    out = load_image(p)
    assert out.size == (10, 20)
    assert out.info["source_format"] == "JPEG"
    p = tmp_path / "x.bmp"
    im.save(p)
    with pytest.raises(ValueError):
        load_image(p)


def test_segment_mock(photo, person):
    class Session:
        def predict(self, im):
            return [Image.fromarray(person)]

    assert np.array_equal(segment(photo, Session()), person)
    keep = np.zeros_like(person)
    keep[0:5] = 255
    m = combine(person, keep, photo.size)
    assert np.all(m[:5] == 255)


@pytest.mark.parametrize(
    "mode", ["Transparente", "Color sólido", "Desenfocado", "Fotografía"]
)
def test_background(photo, person, mode):
    out = replace(
        photo,
        person,
        mode,
        color="#123456",
        photo=Image.new("RGB", (20, 20), "blue"),
        feather=0,
    )
    assert out.size == photo.size
    assert np.array_equal(np.array(out)[50, 50, :3], np.array(photo)[50, 50])
    if mode == "Transparente":
        assert out.getpixel((0, 0))[3] == 0
    if mode == "Color sólido":
        assert out.getpixel((0, 0)) == (18, 52, 86)
    if mode == "Fotografía":
        assert out.getpixel((0, 0)) == (0, 0, 255)


def test_retouch(photo):
    assert skin_mask(photo).shape == (photo.height, photo.width)
    out = retouch(photo, 60, 70)
    assert out.size == photo.size
    assert (
        np.abs(np.asarray(out, dtype=float) - np.asarray(photo, dtype=float)).mean()
        < 30
    )


@pytest.mark.parametrize("amount", [30, -30, 0])
def test_dodge_burn(photo, person, amount):
    out = dodge_burn(photo, person, amount, 0)
    change = (
        np.asarray(out, dtype=float)[40:100, 45:75].mean()
        - np.asarray(photo, dtype=float)[40:100, 45:75].mean()
    )
    assert change * amount > 0 if amount else change == 0
    assert (
        np.abs(
            np.asarray(out, dtype=float)[:5] - np.asarray(photo, dtype=float)[:5]
        ).max()
        <= 2
    )


@pytest.mark.parametrize("channel", ["Todos", *HUES])
def test_hsl(photo, channel):
    out = hsl(photo, channel, 10, 20, 10)
    assert out.size == photo.size
    assert out.mode == "RGB"


def test_white_balance():
    im = Image.new("RGB", (50, 50), (150, 120, 100))
    px = white_balance(im, (25, 25)).getpixel((25, 25))
    assert max(px) - min(px) <= 1
    with pytest.raises(ValueError):
        white_balance(Image.new("RGB", (10, 10)), (2, 2))


@pytest.mark.parametrize("mode", ["Desenfoque", "Pixelado"])
def test_censor(person, mode):
    rng = np.random.default_rng(12)
    im = Image.fromarray(rng.integers(0, 255, (160, 120, 3), dtype=np.uint8))
    out = np.asarray(censor(im, person, mode, 20, 0))
    assert np.array_equal(out[:5], np.asarray(im)[:5])
    assert out[40:100, 45:75].std() < np.asarray(im)[40:100, 45:75].std()


def test_watermark(photo):
    out = watermark(photo, "LAB", 16, 100, "Centro")
    assert not np.array_equal(out, photo)
    assert np.array_equal(watermark(photo, "LAB", 16, 0), photo)


@pytest.mark.parametrize("ratio", list(RATIOS))
def test_transform(photo, ratio):
    out = transform(photo, ratio=ratio)
    if RATIOS[ratio]:
        assert abs(out.width / out.height - RATIOS[ratio]) < 0.03
    assert np.array_equal(transform(photo, flip=True), np.asarray(photo)[:, ::-1])
    assert transform(photo, rotation=90).size == photo.size[::-1]
    assert transform(photo, horizontal=10, vertical=-10).size == photo.size


def test_heal_and_clone(photo, person):
    out = heal(photo, person)
    assert np.array_equal(np.array(out)[:4], np.array(photo)[:4])
    out = clone(photo, (10, 10), (70, 80), 20, 100)
    assert out.getpixel((70, 80)) == photo.getpixel((10, 10))
    assert out.getpixel((0, 0)) == photo.getpixel((0, 0))


def test_editor_alpha(photo):
    layer = Image.new("RGBA", photo.size, (0, 0, 0, 0))
    layer.putpixel((4, 4), (255, 255, 255, 255))
    m = editor_mask({"layers": [layer]}, photo.size)
    assert m.sum() == 1
    with pytest.raises(ValueError):
        editor_mask({"layers": []}, photo.size)


def test_alpha_preserved(photo):
    im = photo.convert("RGBA")
    im.putalpha(123)
    for out in [
        adjust(im, exposure=0.3),
        intelligent(im),
        retouch(im),
        hsl(im, hue=10),
        watermark(im, "", 16, 0),
    ]:
        assert np.array_equal(out.getchannel("A"), im.getchannel("A"))
