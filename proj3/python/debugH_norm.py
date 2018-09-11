import scipy as sp

import numpy as np
from MatchPics import MatchPics
from computeH import computeH

cv = sp.ndimage.imread('../data/cv_cover.jpg')
desk = sp.ndimage.imread('../data/cv_desk.jpg')

[l1,l2] = MatchPics(cv,cv)
H = computeH(l1,l2)
visNorm = np.zeros((l1.shape[0],5))

for i in range(l1.shape[0]):
    estimated = np.array([[l2[i,0]],[l2[i,1]],[1]])
    estimated = np.matmul(H,estimated)
    visNorm[i,0] = l1[i,0]
    visNorm[i,1] = l1[i,1]
    visNorm[i,2] = estimated[0,0]/estimated[2,0]
    visNorm[i,3] = estimated[1,0]/estimated[2,0]
    visNorm[i,4] = np.sqrt((visNorm[i,2]-visNorm[i,0])**2+(visNorm[i,3]-visNorm[i,1])**2)
    
    
c = 2