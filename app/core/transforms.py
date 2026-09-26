from PIL import Image, ImageOps
import cv2
import numpy as np

RATIOS = {
    "Original": None,
    "1:1": 1,
    "4:5": 4 / 5,
    "3:4": 3 / 4,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "2:3": 2 / 3,
}


def transform(im, rotation=0, flip=False, ratio="Original", horizontal=0, vertical=0):
    out = im.copy()
    if horizontal or vertical:
        w, h = out.size
        dx, dy = (
            np.clip(horizontal, -20, 20) / 100 * w,
            np.clip(vertical, -20, 20) / 100 * h,
        )
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32(
            [
                [max(dx, 0), max(dy, 0)],
                [w - 1 - max(-dx, 0), max(-dy, 0)],
                [w - 1 - max(dx, 0), h - 1 - max(-dy, 0)],
                [max(-dx, 0), h - 1 - max(dy, 0)],
            ]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        out = Image.fromarray(
            cv2.warpPerspective(
                np.array(out), matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101
            )
        )
    if rotation:
        out = out.rotate(
            float(rotation), resample=Image.Resampling.BICUBIC, expand=True
        )
    if flip:
        out = ImageOps.mirror(out)
    r = RATIOS[ratio]
    if r:
        w, h = out.size
        nw, nh = (int(h * r), h) if w / h > r else (w, int(w / r))
        out = out.crop(((w - nw) // 2, (h - nh) // 2, (w + nw) // 2, (h + nh) // 2))
    out.info.update(im.info)
    return out
