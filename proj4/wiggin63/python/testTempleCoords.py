import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import scipy.ndimage 

from eightpoint import eightpoint
from LineEq import LineEq
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate

def dist(x1,x2):
    d = (x1[0]-x2[0])**2 + (x1[1]-x2[1])**2
    return d

def reProjError(P1,x1,P2,x2,X):
    
    error = 0
    cnt = x1.shape[0]
    x1h = np.matmul(P1,np.transpose(X))
    x2h = np.matmul(P2,np.transpose(X))
    x1h = x1h/x1h[2,:]
    x2h = x2h/x2h[2,:]
    x1h = x1h[0:2,:]
    x2h = x2h[0:2,:]

    x1h = x1h.T
    x2h = x2h.T
    for i in range(cnt):
        error = error + dist(x1[i,:],x1h[i,:])
        error = error + dist(x2[i,:],x2h[i,:])

    
    return error
dictionary = np.load('../data/some_corresp.npz')
#points are (x,y)
pts1 = dictionary['pts1']
pts2 = dictionary['pts2']
#Load images 
im1 = sp.ndimage.imread('../data/im1.png')
im2 = sp.ndimage.imread('../data/im2.png')
im1 = im1.astype(np.float)/im1.max()
im2 = im2.astype(np.float)/im2.max()
im1 = sp.ndimage.rotate(im1,angle=-90)
im2 = sp.ndimage.rotate(im2,angle=-90)
#Finding Fundamental 
M = max(pts1.shape)
F = eightpoint(pts1,pts2,M)
#Finding Coorespondances
coords = np.load('../data/templeCoords.npz')
x1 = coords['pts']
x2 = epipolarCorrespondence(im1,im2,F,x1)
#Finding E
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
K1 = np.transpose(K1)
K2 = np.transpose(K2)
E = essentialMatrix(F, K1, K2)
#Find Camera Projection Matrix 
Rt1 = np.concatenate((np.identity(3),np.zeros((3,1))),axis=1) #Assuming not rotation or translation in first image
P1 = np.matmul(K1,Rt1)
Rt2s = camera2(E)
P2s = np.zeros(Rt2s.shape)
Xs = np.zeros((x1.shape[0],4,4))                    #x y z 
depth_cnt = np.zeros((4))
dets = np.zeros((4))

for i in range(4):
    Rt2s[:,3,i] = Rt2s[:,3,i]/Rt2s[2,3,i]
    P2s[:,:,i] = np.matmul(K2,Rt2s[:,:,i])
    Xs[:,:,i] = triangulate(P1,x1,P2s[:,:,i],x2)    
    depth_cnt[i] = np.sum(np.greater(Xs[:,2,i],0))  #looks just at Z
    dets[i] = np.linalg.det(Rt2s[:,0:3,i])
correct_P2 = np.argmax(depth_cnt)
#correct_P2 = 1
X = Xs[:,:,correct_P2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c='r', marker='o')
#plt.scatter(X[:,1],X[:,0])
plt.show()
error = reProjError(P1,x1,P2s[:,:,correct_P2],x2,X)