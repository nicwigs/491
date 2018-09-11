# createFilterBank.py

# Vishnu Naresh Boddeti | CSE-491 HW2
#
# creates a bank of filters to be used as
# descriptors for points in the image

import math
import numpy as np
from scipy import ndimage

def gaussian(shape=(3, 3), sigma=0.5):
    """
    2D gaussian filter
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def laplacian_of_gaussian(shape=(3, 3), sigma=0.5):
    """
    2D laplacian of gaussian filter
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    tmp1 = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    tmp2 = 1 - (x * x + y * y) / (2. * sigma * sigma)
    tmp = tmp1 * tmp2
    h = -tmp / (np.pi * math.pow(sigma, 4))

    return h


def createFilterBank():

    # Code to generate reasonable filter bank
    gaussianScales = [1, 2, 4, 8, math.sqrt(2) * 8]
    logScales = [1, 2, 4, 8, math.sqrt(2) * 8]
    dxScales = [1, 2, 4, 8, math.sqrt(2) * 8]
    dyScales = [1, 2, 4, 8, math.sqrt(2) * 8]

    filterBank = []

    for scale in gaussianScales:
        siz = 2 * math.ceil(scale * 2.5) + 1
        tmp = gaussian(shape=(siz, siz), sigma=scale)
        filterBank.append(tmp)

    for scale in logScales:
        siz = 2 * math.ceil(scale * 2.5) + 1
        tmp = laplacian_of_gaussian(shape=(siz, siz), sigma=scale)
        filterBank.append(tmp)

    xgrad = np.array([[-1, 0, 1]])
    for scale in dxScales:
        siz = 2 * math.ceil(scale * 2.5) + 1
        f = gaussian(shape=(siz, siz), sigma=scale)
        f = ndimage.correlate(f, xgrad)
        filterBank.append(f)

    for scale in dyScales:
        f = gaussian(shape=(siz, siz), sigma=scale)
        f = ndimage.correlate(f, xgrad.transpose())
        filterBank.append(f)

    return filterBank
