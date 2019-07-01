# myEdgeFilter.py

import math
import numpy as np
from myImageFilter import myImageFilter


def gaussian(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
    

def myEdgeFilter(img, sigma):

    shape=(3, 3)
    sobel_y = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    kernel = gaussian(shape,sigma)
    img = myImageFilter(img,kernel) #Gaussian Blur
    Ix = myImageFilter(img,sobel_x) 
    Iy = myImageFilter(img,sobel_y)       
    img = np.sqrt(np.square(Ix)+np.square(Iy))
    
    #NonMaximum Suppresion
    row,col = img.shape
    theta = np.arctan(Iy/Ix)
    
    for i in range(1,row-1):
        for j in range(1,col-1):         
            if theta[i,j] < -3*math.pi/8 or theta[i,j] > 3*math.pi/8:  #gradient: x = 0
                if (img[i+1,j] or img[i-1,j]) > img[i,j] :
                    img[i,j] = 0
            elif theta[i,j] < -math.pi/8:                         #gradient y = -x
                if (img[i+1,j-1] or img[i-1,j+1]) > img[i,j]:
                    img[i,j] = 0
            elif theta[i,j] < math.pi/8:                          #gradient y = 0
                if (img[i,j-1] or img[i,j+1]) > img[i,j]:
                    img[i,j] = 0
            elif theta[i,j] < 3*math.pi/8:                        #gradient y = x
                if (img[i-1,j-1] or img[i+1,j+1]) > img[i,j]:
                    img[i,j] = 0
    
    return img

#a = np.ones((5,5))
#sigma = 0.6

#im = myEdgeFilter(a,sigma)
