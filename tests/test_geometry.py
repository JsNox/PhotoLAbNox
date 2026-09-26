import numpy as np
import pytest
from PIL import Image
from app.core.body_reshape import (
    estimate,
    calibrate,
    reshape,
    fields,
    effective,
    overlay,
)
from app.core.liquify import LiquifyState


def test_five_regions(person, photo):
    r = estimate(person)
    assert len(r) == 5
    assert all(x.left == 35 and x.right == 84 and x.width == 50 for x in r)
    manual = calibrate(person, [(55, y) for y in [40, 60, 80, 100, 130]])
    assert [x.y for x in manual] == [40, 60, 80, 100, 130]
    assert overlay(photo, manual).size == photo.size
    with pytest.raises(ValueError):
        calibrate(person, [(50, 40)] * 5)
    with pytest.raises(ValueError):
        calibrate(person, [(50, 40)])


def test_expand_bust_and_background(photo, person):
    r = estimate(person)
    out, mask, amounts = reshape(photo, person, r, bust=20, intensity=100)
    y = r[1].y
    assert np.count_nonzero(mask[y] > 127) > np.count_nonzero(person[y] > 127)
    assert amounts[0] == 0.18
    assert np.array_equal(np.array(out)[:10], np.array(photo)[:10])
    assert np.array_equal(np.array(out)[:, :20], np.array(photo)[:, :20])
    _, smaller, _ = reshape(photo, person, r, waist=-20, intensity=100)
    assert np.count_nonzero(smaller[r[2].y] > 127) < np.count_nonzero(
        person[r[2].y] > 127
    )


def test_independent_inclined_centers():
    mask = np.zeros((200, 200), np.uint8)
    for y in range(10, 195):
        mask[y, 20 + y // 3 : 80 + y // 3] = 255
    r = estimate(mask)
    assert r[1].center < r[2].center < r[3].center
    mx, my = fields((200, 200), r, effective(20, -20, 20, 20, 100))
    for region in r:
        x = round(region.center)
        assert abs(mx[region.y, x] - x) < 1
    assert np.max(abs(effective(20, -20, 20, 20, 100))) <= 0.18


def test_liquify_restore(photo, person):
    state = LiquifyState.create(photo, person)
    out, m = state.apply((55, 80), (70, 80), 40, 25, top=40)
    assert not np.array_equal(out, photo)
    assert np.array_equal(np.array(out)[:30], np.array(photo)[:30])
    assert np.array_equal(np.array(out)[:, :10], np.array(photo)[:, :10])
    before = abs(state.dx).sum()
    state.apply((55, 80), radius=50, strength=100, restore=True, top=40)
    assert abs(state.dx).sum() < before


@pytest.mark.parametrize("size", [(64, 96), (200, 100), (513, 769)])
def test_resolutions(size):
    im = Image.new("RGB", size, (120, 90, 60))
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    mask[h // 10 : h * 9 // 10, w // 3 : w * 2 // 3] = 255
    out, m, _ = reshape(im, mask, bust=10, waist=-10, hip=5)
    assert out.size == size and m.shape == (h, w)
