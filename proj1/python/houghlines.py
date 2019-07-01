# houghlines.py


import math
import numpy as np


def houghpixels(f, theta, rho, rbin, cbin):
    y, x = np.where(f)
    x = x - 1
    y = y - 1
    theta_c = theta[int(cbin)] * math.pi / 180
    rho_xy = x * math.cos(theta_c) + y * math.sin(theta_c)
    nrho = len(rho)
    slope = (nrho - 1) / (rho[-1] - rho[0])
    rho_bin_index = np.round(slope * (rho_xy - rho[0]))

    idx = (rho_bin_index == np.round(rbin)[0])
    r = x[idx] + 1
    c = y[idx] + 1
    return r, c


def houghlines(f, theta, rho, rr, cc, fillgap=20, minlength=40):
    numlines = 0
    lines = []

    for k in range(len(rr)):
        rbin = rr[k]
        cbin = cc[k]
        r, c = houghpixels(f, theta, rho, rbin, cbin)

        omega = (90 - theta[int(cbin)]) * math.pi / 180
        T = np.array([[math.cos(omega), math.sin(omega)], [-math.sin(omega), math.cos(omega)]])

        r = np.expand_dims(r - 1, axis=1)
        c = np.expand_dims(c - 1, axis=1)
        temp = np.concatenate((r, c), axis=1)

        xy = np.matmul(temp, T)
        x = np.sort(xy[:, 0])

        diff_x = np.concatenate((np.expand_dims(np.diff(x), axis=1), np.array([[np.Inf]]))) 
        temp = np.where(diff_x > fillgap)
        idx = np.concatenate((np.array([[-1]]), np.expand_dims(temp[0], axis=1)))

        for p in range(0, idx.size - 1):
            x1 = x[idx[p] + 1][0]
            x2 = x[idx[p + 1]][0]
            linelength = x2 - x1
            if linelength > minlength:
                point1 = np.array([[x1, rho[int(rbin)]]])
                point2 = np.array([[x2, rho[int(rbin)]]])
                Tinv = np.linalg.inv(T)
                point1 = np.matmul(point1, Tinv)
                point2 = np.matmul(point2, Tinv)
                numlines += 1
                line = {}
                line['point1'] = point1 + 1
                line['point2'] = point2 + 1
                line['length'] = linelength
                line['theta'] = theta[int(cbin)]
                line['rho'] = rho[int(rbin)]
                lines.append(line)

    return lines
