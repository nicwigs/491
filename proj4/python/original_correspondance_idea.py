# epipolarCorrespondence.py
from skimage.feature import BRIEF,corner_fast,corner_peaks,match_descriptors
from skimage.color import rgb2gray
import numpy as np
from LineEq import LineEq

def findPoints(peaks,im2,F,im1_point,threshold):
    
    point_homog = [im1_point[0],im1_point[1],1]     #image 1 point in homogonous coords
    line = np.matmul(F,point_homog)                 #Epipolar Line coefficients 
    
    close = np.zeros((peaks.shape[0],1))            #1 if point close enough to line

    for i in range(peaks.shape[0]):                 #through all the peaks it found
        x = peaks[i,0]
        y = peaks[i,1]
        line_y = LineEq(line,x)                     #get y estimate from F
        dist = abs(line_y-y)
        close[i] = (dist < threshold)               #1 if close enough
        
    indx = close.nonzero()[0]                       #Indecies of the close points 
    points = np.take(peaks,indx,axis=0)             #points from image1 that are in close enough
    
    return points


def epipolarCorrespondence(im1,im2,F,x1):
    #x1 comes in (x,y)
    threshold = 50
    
    if im1.ndim == 3:
        im1 = rgb2gray(im1)   
    if im2.ndim == 3:
        im2 = rgb2gray(im2)
    #Finds and refines peaks that will look for matches
    peaks = corner_peaks(corner_fast(im2,n=12,threshold=0.15),min_distance=1) #get corners
    for i in range(x1.shape[0]):
        x = x1[i,0]
        y = x1[i,1]
        points = findPoints(peaks,im2,F,x1[i,:],threshold) #points is (y,x)

    extractor = BRIEF()
    extractor.extract(im1,np.array([x1[0,1],x1[0,0]])#Point from im1 into y,x to match other points
    descriptors1 = extractor.descriptors
    
    extractor.extract(im2,points)
    descriptors2 = extractor.descriptors
    
    matches = match_descriptors(descriptors1,descriptors2,metric = 'hamming',cross_check=True)
    
    locs1 = points1[matches[:,0]]
    locs2 = points2[matches[:,1]]
    #Change to x,y (col,row)
    xy1 = np.array([locs1[:,1],locs1[:,0]])
    xy1 = xy1.transpose()
    
    
    
    
    
    
    return x2
