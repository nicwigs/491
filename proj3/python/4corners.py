import scipy as sp
import numpy as np

from skimage.transform import warp,ProjectiveTransform,resize
import matplotlib.pyplot as plt

from MatchPics import MatchPics
from computeH import computeH
from computeH_norm import computeH_norm
from computeH_ransac import computeH_ransac



desk = sp.ndimage.imread('../data/cv_desk.jpg')
cv = sp.ndimage.imread('../data/cv_cover.jpg')
hp = sp.ndimage.imread('../data/hp_cover.jpg')

#hp_corners = np.array([[199,0],[1,1],[0,294],[199,294]])
#x,y (col,row)
cv_corners = np.array([[0,0],[350,0],[0,440],[350,440]])
desk_corners = np.array([[237,192],[490,192],[150,485],[575,480]])
#These are oposite these y,x
#cv_corners = np.array([[0,0],[0,350],[440,0],[440,350]])
#desk_corners = np.array([[192,237],[192,490],[485,150],[480,575]])

H = computeH_norm(desk_corners,cv_corners)
#H = computeH(cv_corners,desk_corners)

correct = ProjectiveTransform()
correct.estimate(src=cv_corners,dst=desk_corners)
H_correct = correct.params


vis4 = np.zeros((cv_corners.shape[0],5))

for i in range(cv_corners.shape[0]):
    estimated = np.array([[cv_corners[i,0]],[cv_corners[i,1]],[1]])
    print(estimated[0,0],estimated[1,0],estimated[2,0])
    estimated = np.matmul(H,estimated)
    print(estimated[0,0],estimated[1,0],estimated[2,0])
    print(estimated[0,0]/estimated[2,0],estimated[1,0]/estimated[2,0],estimated[2,0]/estimated[2,0])
    print(desk_corners[i,0],desk_corners[i,1])
    print('--')
    vis4[i,0] = desk_corners[i,0]
    vis4[i,1] = desk_corners[i,1]
    vis4[i,2] = estimated[0,0]/estimated[2,0]
    vis4[i,3] = estimated[1,0]/estimated[2,0]
    vis4[i,4] = np.sqrt((vis4[i,2]-vis4[i,0])**2+(vis4[i,3]-vis4[i,1])**2)


hp_scale = resize(hp,cv.shape)
#H_inverse = np.linalg.inv(H)
H_transform = ProjectiveTransform(H)

H_transform2 = ProjectiveTransform(H_correct)
#.inverse
warped = warp(hp,H_transform,output_shape=desk.shape[0:2], cval=-1)
plt.imshow(warped)