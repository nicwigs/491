# myEdgeFilter.py

import math
import numpy as np
from myImageFilter import myImageFilter


def gaussian(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def myEdgeFilter(img, sigma):

    return img
