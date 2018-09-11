import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage 
from eightpoint import eightpoint
from LineEq import LineEq
from epipolarCorrespondence import epipolarCorrespondence

dictionary = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')
#points are (x,y)
pts1 = dictionary['pts1']
pts2 = dictionary['pts2']

im1 = sp.ndimage.imread('../data/im1.png')
im2 = sp.ndimage.imread('../data/im2.png')
im1 = im1.astype(np.float)/im1.max()
im2 = im2.astype(np.float)/im2.max()
im1 = sp.ndimage.rotate(im1,angle=-90)
im2 = sp.ndimage.rotate(im2,angle=-90)

M = max(pts1.shape)
F = eightpoint(pts1,pts2,M)

#Finding Coorespondances
coords = np.load('../data/templeCoords.npz')
x1 = coords['pts']
x2 = epipolarCorrespondence(im1,im2,F,x1)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)         #set up plot
ax1.set_title('Image 1 Points')
ax2.set_title('Line and Estimated')
for i in range(0,110,5):  
    point1 = x1[i,:]
    pointhp = [point1[0],point1[1],1]
    line = np.matmul(F,pointhp)  
    point2 = x2[i,:]
    x = range(im1.shape[0])                                 #x range for line
    y = LineEq(line,x)

    ax1.scatter(point1[1],point1[0])                        #on image y,x
    ax2.plot(x,y)
    ax2.scatter(point2[1],point2[0],marker='^')

ax1.imshow(im1)
ax2.imshow(im2)

plt.show() 