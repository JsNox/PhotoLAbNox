import numpy as np
from app.models.model_manager import segmentation_session
from app.utils.image_utils import mask_array


def segment(im, session=None):
    session = session if session is not None else segmentation_session()
    masks = session.predict(im.convert("RGB"))
    return np.uint8(mask_array(masks[0], im.size) * 255)


def combine(person, keep, size):
    a = mask_array(person, size)
    if keep is not None:
        a = np.maximum(a, mask_array(keep, size))
    return np.uint8(a * 255)
