# drawLine.py

import numpy as np


def drawLine(img, begin, end):

    delta = 0.05 / np.linalg.norm(end - begin)
    t = np.arange(0, 1, delta)

    x = begin[0][0] + t * (end[0][0] - begin[0][0])
    y = begin[0][1] + t * (end[0][1] - begin[0][1])

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2).repeat(3, axis=2)

    if img.dtype != np.float64:
        img = np.float64(img) / 255

    x = np.round(x)
    y = np.round(y)

    x = np.clip(x, 0, img.shape[1] - 1)
    y = np.clip(y, 0, img.shape[0] - 1)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    r[np.int64(y), np.int64(x)] = 0.0
    g[np.int64(y), np.int64(x)] = 1.0
    b[np.int64(y), np.int64(x)] = 0.0

    r = np.expand_dims(r, axis=2)
    g = np.expand_dims(g, axis=2)
    b = np.expand_dims(b, axis=2)

    img = np.concatenate((r, g, b), axis=2)
    return img
