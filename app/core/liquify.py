from dataclasses import dataclass
import numpy as np
from app.core.body_reshape import composite_warp
from app.core.background import clean_plate
from app.utils.image_utils import mask_array


@dataclass
class LiquifyState:
    base: object
    mask: object
    plate: object
    dx: object
    dy: object

    @classmethod
    def create(cls, im, mask):
        shape = (im.height, im.width)
        return cls(
            im.copy(),
            np.uint8(mask_array(mask, im.size) * 255),
            clean_plate(im, mask),
            np.zeros(shape, np.float32),
            np.zeros(shape, np.float32),
        )

    def apply(
        self, origin, destination=None, radius=90, strength=20, restore=False, top=0
    ):
        yy, xx = np.mgrid[: self.base.height, : self.base.width].astype(np.float32)
        ox, oy = origin
        radius = max(2, float(radius))
        dist = ((xx - ox) ** 2 + (yy - oy) ** 2) / radius**2
        weight = np.maximum(1 - dist, 0) ** 3
        weight *= np.array(self.mask) / 255
        weight[yy < top] = 0
        gain = np.clip(strength / 100, 0, 1)
        if restore:
            self.dx *= 1 - weight * gain
            self.dy *= 1 - weight * gain
        else:
            if destination is None:
                raise ValueError("Marca el destino del arrastre.")
            delta = np.array(destination, dtype=float) - np.array(origin, dtype=float)
            length = np.linalg.norm(delta)
            delta *= min(1, radius * 0.5 / max(length, 1)) * gain
            self.dx -= weight * delta[0]
            self.dy -= weight * delta[1]
        limit = min(self.base.size) * 0.18
        self.dx = np.clip(self.dx, -limit, limit)
        self.dy = np.clip(self.dy, -limit, limit)
        return composite_warp(
            self.base, self.mask, xx + self.dx, yy + self.dy, self.plate
        )
