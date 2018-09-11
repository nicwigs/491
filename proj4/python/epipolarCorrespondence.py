# epipolarCorrespondence.py
from skimage.color import rgb2gray
import numpy as np
from LineEq import LineEq
def dist(w1,w2):
    #input windows fomr im1 and im2
    sz = w1.size
    w1 = w1.reshape((sz))
    w2 = w2.reshape((sz))
    d = 0
    for i in range(sz):                  #calc euclidian dist
        d = d + (w1[i] - w2[i])**2
    d = np.sqrt(d)
    return d

def findPoint(pad1,pad2,lineco,hp,pad):
    delta = 9 #how many points to check above and below the line
    #pad of image 1 and 2, line coefficents, homogeneous point from image1
    x = np.arange(pad1.shape[1])          #x of epipolar line
    y = LineEq(lineco,x)                  #y of epipolar line
    y = y.astype(int)
    hp = hp.astype(int)
    points = x.size
    point = np.array([0,0])
    mindist = 10000000000
    w1 = pad1[hp[0]+pad-pad:hp[0]+pad+pad+1,hp[1]+pad-pad:hp[1]+pad+pad+1] #weird since pad 

    for i in range(pad,points-pad): #start at one, end one short since this is pad
        for j in range(-delta,delta):
            if pad <= (y[i]+j) < (pad1.shape[0]-2*pad):
                w2 = pad2[y[i]+j-pad:y[i]+j+pad+1,x[i]-pad:x[i]+pad+1]
                d = dist(w1,w2)
                if d < mindist:
                    point = np.array([y[i-pad]+j,x[i-pad]])
                    mindist = d
            
    return point

def epipolarCorrespondence(im1,im2,F,x1):
    #x1 is array of points from im1 comes in (x,y)
    
    if im1.ndim == 3:
        im1 = rgb2gray(im1)   
    if im2.ndim == 3:
        im2 = rgb2gray(im2)
    w = 21 #window size
    pad = w//2
    pad1 = np.pad(im1,pad,'constant',constant_values = 0)
    pad2 = np.pad(im2,pad,'constant',constant_values = 0)
    
    x2 = np.zeros(x1.shape)
    sz = x1.shape[0]
    for i in range(sz):                     #points in correspondance 
        point = np.array([x1[i,0],x1[i,1],1]) #point of im1 in homogenous coords
        lineco = np.matmul(F,point)           #line in img2 the point should be
        
        x2[i,:] = findPoint(pad1,pad2,lineco,point,pad)

    return x2
