# RGB2Lab.py

# Vishnu Boddeti | CSE-491 HW2
#
# function [L, a, b] = RGB2Lab(R, G, B)
# RGB2Lab takes matrices corresponding to Red, Green, and Blue, and
# transforms them into CIELab.  This transform is based on ITU-R
# Recommendation  BT.709 using the D65 white point reference.
# The error in transforming RGB -> Lab -> RGB is approximately
# 10^-5.  RGB values can be either between 0 and 1 or between 0 and 255.

import numpy as np


def RGB2Lab(R, G=None, B=None):

    if G is None and B is None:
        B = np.float64(R[:, :, 2])
        G = np.float64(R[:, :, 1])
        R = np.float64(R[:, :, 0])

    if np.max(R) > 1.0 or np.max(G) > 1.0 or np.max(B) > 1.0:
        R = R / 255.0
        G = G / 255.0
        B = B / 255.0

    M, N = R.shape
    s = M * N

    # Set a threshold
    T = 0.008856
    RGB = np.concatenate((np.reshape(R, (1, s)), np.reshape(G, (1, s)),
                         np.reshape(B, (1, s))), axis=0)
    MAT = np.array([[0.412453, 0.357580, 0.180423],
                   [0.212671, 0.715160, 0.072169],
                   [0.019334, 0.119193, 0.950227]]
                   )
    XYZ = np.matmul(MAT, RGB)

    X = XYZ[0, :] / 0.950456
    Y = XYZ[1, :]
    Z = XYZ[2, :] / 1.088754

    XT = X > T
    YT = Y > T
    ZT = Z > T

    fX = XT * np.power(X, 1.0 / 3.0) + (~XT) * (7.787 * X + 16.0 / 116.0)

    # Compute L
    Y3 = np.power(Y, 1.0 / 3.0)
    fY = YT * Y3 + (~YT) * (7.787 * Y + 16.0 / 116.0)
    L = YT * (116 * Y3 - 16.0) + (~YT) * (903.3 * Y)
    fZ = ZT * np.power(Z, 1.0 / 3.0) + (~ZT) * (7.787 * Z + 16 / 116)

    # Compute a and b
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    L = np.reshape(L, [M, N])
    a = np.reshape(a, [M, N])
    b = np.reshape(b, [M, N])

    L = np.stack([L, a, b], axis=2)

    return L
