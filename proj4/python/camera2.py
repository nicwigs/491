# camera2.py

import numpy as np


def camera2(E):
    U, S, V = np.linalg.svd(E)
    m = (S[0] + S[1]) / 2
    R = np.array([[m, 0, 0], [0, m, 0], [0, 0, 0]])
    E = np.dot(np.dot(U, R), V)
    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    if np.linalg.det(np.dot(np.dot(U, W), V)) < 0:
        W = -W

    M2s = np.zeros((3, 4, 4))
    M2s[:, 0:3, 0] = np.dot(U, np.dot(W, V))
    M2s[:, 3, 0] = U[:, 2] / max(abs(U[:, 2]))

    M2s[:, 0:3, 1] = np.dot(U, np.dot(W, V))
    M2s[:, 3, 1] = -U[:, 2] / max(abs(U[:, 2]))

    M2s[:, 0:3, 2] = np.dot(U, np.dot(W.transpose(), V))
    M2s[:, 3, 2] = U[:, 2] / max(abs(U[:, 2]))

    M2s[:, 0:3, 3] = np.dot(U, np.dot(W.transpose(), V))
    M2s[:, 3, 3] = -U[:, 2] / np.max(np.abs(U[:, 2]))

    return M2s