import scipy as sp
from skimage.transform import warp,ProjectiveTransform
import matplotlib.pyplot as plt

import numpy as np
from MatchPics import MatchPics
from computeH import computeH

cv = sp.ndimage.imread('../data/cv_cover.jpg')
desk = sp.ndimage.imread('../data/cv_desk.jpg')

[l1,l2] = MatchPics(cv,cv)
H = computeH(l1,l2)
vis = np.zeros((l1.shape[0],5))

for i in range(l1.shape[0]):
    estimated = np.array([[l2[i,0]],[l2[i,1]],[1]])
    estimated = np.matmul(H,estimated)
    vis[i,0] = l1[i,0]
    vis[i,1] = l1[i,1]
    vis[i,2] = estimated[0,0]/estimated[2,0]
    vis[i,3] = estimated[1,0]/estimated[2,0]
    vis[i,4] = np.sqrt((vis[i,2]-vis[i,0])**2+(vis[i,3]-vis[i,1])**2)
    
    
c = 2

H_transform = ProjectiveTransform(H)
#548,731
warped = warp(cv,H_transform,output_shape=(439,349))
plt.imshow(warped)