import scipy as sp
import numpy as np
from skimage.transform import warp,ProjectiveTransform, resize
import matplotlib.pyplot as plt

from MatchPics import MatchPics
from computeH import computeH
from computeH_norm import computeH_norm
from computeH_ransac import computeH_ransac

desk = sp.ndimage.imread('../data/cv_desk.jpg')
cv = sp.ndimage.imread('../data/cv_cover.jpg')
hp = sp.ndimage.imread('../data/hp_cover.jpg')
hp_scale = resize(hp,cv.shape)

[desk_points,cv_points] = MatchPics(desk,cv)

iterations = 20000
threshold = 1
[H,i] = computeH_ransac(cv_points,desk_points,iterations,threshold)

H_transform = ProjectiveTransform(H)
warped = warp(hp_scale,H_transform.inverse,output_shape=desk.shape[0:2], cval=-1)

desk_mask = np.ma.masked_equal(warped,-1).mask
cover_mask = np.ma.masked_not_equal(warped,-1).mask

warped = 255*warped
warped = warped.astype(np.uint8)

final = np.add(np.multiply(desk_mask,desk),np.multiply(cover_mask,warped))
plt.imshow(final)