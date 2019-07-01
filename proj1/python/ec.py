# ec.py

import os
import math
import numpy as np
import scipy as sp
from scipy import ndimage

from drawLine import drawLine
from houghlines import houghlines
from myHoughLines import myHoughLines
from myEdgeFilter import myEdgeFilter
from myHoughTransform import myHoughTransform


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


datadir = '../ec'
resultsdir = '../ec'

# parameters
sigma = 1.5
threshold = 0.3
rhoRes = 2.0
thetaRes = math.pi / 180
nLines = 50
# end of parameters

imglist = []
for file in os.listdir(datadir):
    if file.endswith(".jpg"):
        imglist.append(file)

imglist.sort()
for i in range(len(imglist)):
    # read in images
    fname = os.path.join(datadir, imglist[i])
    img = sp.ndimage.imread(fname)
    if img.ndim == 3:
        img = rgb2gray(img)

    img = np.float64(img) / 255.0

    # actual Hough line code function calls
    Im = myEdgeFilter(img, sigma)
    H, rhoScale, thetaScale = myHoughTransform(Im, threshold, rhoRes, thetaRes)
    rhos, thetas = myHoughLines(H, nLines)
    thresh_img = Im > threshold
    lines = houghlines(thresh_img, 180 * (thetaScale / math.pi), rhoScale,
                       rhos, thetas, fillgap=5, minlength=20)

    # everything below here just saves the outputs to files
    filename_w_ext = os.path.basename(imglist[i])
    filename, file_extension = os.path.splitext(filename_w_ext)
    fname = os.path.join(resultsdir, filename + '_01edge.png')
    temp = np.uint8(255.0 * np.sqrt(Im / np.max(Im)))
    sp.misc.imsave(fname, temp)
    fname = os.path.join(resultsdir, filename + '_02threshold.png')
    temp = np.uint8(255.0 * (Im > threshold))
    sp.misc.imsave(fname, temp)
    fname = os.path.join(resultsdir, filename + '_03hough.png')
    sp.misc.imsave(fname, H / np.max(H))
    fname = os.path.join(resultsdir, filename + '_04lines.png')
    img2 = img
    for j in range(len(lines)):
        img2 = drawLine(img2, lines[j]['point1'], lines[j]['point2'])
    temp = np.uint8(255.0 * img2)
    sp.misc.imsave(fname, temp)
