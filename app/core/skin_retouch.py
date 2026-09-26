import cv2
import numpy as np
from app.utils.image_utils import rgb, result_like, blend


def skin_mask(im):
    a = rgb(im)
    ycc = cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    one = cv2.inRange(ycc, (0, 130, 75), (255, 180, 135))
    two = cv2.inRange(hsv, (0, 20, 25), (25, 220, 255))
    return cv2.GaussianBlur(cv2.bitwise_and(one, two), (0, 0), 2)


def retouch(im, intensity=35, texture=70):
    if intensity == 0:
        return im.copy()
    a = rgb(im)
    smooth = cv2.bilateralFilter(a, 9, 35, 9)
    clean = cv2.medianBlur(smooth, 3)
    detail = a.astype(np.float32) - cv2.GaussianBlur(a, (0, 0), 1).astype(np.float32)
    target = result_like(
        im, clean.astype(np.float32) + detail * np.clip(texture / 100, 0, 1)
    )
    return blend(
        im,
        target,
        skin_mask(im).astype(np.float32) / 255 * np.clip(intensity / 100, 0, 1),
    )
