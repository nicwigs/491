import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage 
from eightpoint import eightpoint
from LineEq import LineEq

dictionary = np.load('../data/some_corresp.npz')
keys = dictionary.keys()

intrinsics = np.load('../data/intrinsics.npz')
key2 = intrinsics.keys()

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

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)         #set up plot
ax1.set_title('Epipolar Point')
ax2.set_title('Epipolar Line')

for i in range(0,110,7):  
    point = [pts1[i,0],pts1[i,1],1]
    line = np.matmul(F,point) 
    ax1.scatter(point[1],point[0])                          #on image y,x
    x = range(im1.shape[0])                                 #x range for line
    y = LineEq(line,x)
    ax2.plot(x,y)#,x)
    
ax1.imshow(im1)
ax2.imshow(im2)