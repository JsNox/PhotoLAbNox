import numpy as np
import pytest
from PIL import Image
from app.utils.paths import setup

setup()


@pytest.fixture
def photo():
    y, x = np.mgrid[:160, :120]
    a = np.stack([80 + x, 55 + y // 2, 50 + x // 3 + y // 4], axis=-1).astype(np.uint8)
    return Image.fromarray(a)


@pytest.fixture
def person():
    m = np.zeros((160, 120), np.uint8)
    m[12:145, 35:85] = 255
    return m
