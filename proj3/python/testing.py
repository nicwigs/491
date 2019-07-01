#Testing
import scipy as sp
import numpy as np


from MatchPics import MatchPics
from computeH import computeH
from computeH_norm import computeH_norm
from computeH_ransac import computeH_ransac



I1 = sp.ndimage.imread('../data/cv_desk.jpg')
I2 = sp.ndimage.imread('../data/cv_cover.jpg')

[l1,l2] = MatchPics(I1,I2)

#H = computeH(l1,l2)
#l1_estimate = np.matmul(H,)
#H = computeH_norm(l1,l2)

#[bestH2to1, inliers] = computeH_ransac(l1,l2)
