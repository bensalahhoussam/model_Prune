#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return [f[:-4] for f in os.listdir(dir)]


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return [(id, i) for id in ids for i in range(n)]


def to_cropped_imgs(images, scale):
    """From a list of tuples, returns the correct cropped img"""
    for img in images :

        im = resize_and_crop(Image.open(img), scale=scale)
        yield get_square(im, 0)

def get_imgs_and_masks(images,masks, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(images, scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(masks, scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
