# epipolarCorrespondence.py
from skimage.color import rgb2gray
import numpy as np
from LineEq import LineEq
def dist(w1,w2):
    #input windows fomr im1 and im2
    w1 = w1.reshape((9))
    w2 = w2.reshape((9))
    d = 0
    for i in range(9):                  #calc euclidian dist
        d = d + (w1[i] - w2[i])**2
    d = np.sqrt(d)
    return d

def findPoint(pad1,pad2,lineco,hp):
    #pad of image 1 and 2, line coefficents, homogeneous point from image1
    x = np.arange(pad1.shape[1])          #x of epipolar line
    y = LineEq(lineco,x)                  #y of epipolar line
    y = y.astype(int)
    hp = hp.astype(int)
    points = x.size
    point = np.array([0,0])
    mindist = 10000000000
    for i in range(1,points-1): #start at one, end one short since this is pad
        w1 = pad1[hp[0]+1-1:hp[0]+1+2,hp[1]+1-1:hp[1]+1+2] #weird since pad 
        w2 = pad2[y[i]-1:y[i]+2,x[i]-1:x[i]+2]
        d = dist(w1,w2)
        if d < mindist:
            point = np.array([y[i-1],x[i-1]])
            mindist = d
            
            
    return point

def epipolarCorrespondence(im1,im2,F,x1):
    #x1 is array of points from im1 comes in (x,y)
    
    if im1.ndim == 3:
        im1 = rgb2gray(im1)   
    if im2.ndim == 3:
        im2 = rgb2gray(im2)
    w = 
    pad1 = np.pad(im1,1,'constant',constant_values = 0)
    pad2 = np.pad(im2,1,'constant',constant_values = 0)
    
    x2 = np.zeros(x1.shape)
    sz = x1.shape[0]
    for i in range(sz):                     #points in correspondance 
        point = np.array([x1[i,0],x1[i,1],1]) #point of im1 in homogenous coords
        lineco = np.matmul(F,point)           #line in img2 the point should be
        
        x2[i,:] = findPoint(pad1,pad2,lineco,point)

    return x2

