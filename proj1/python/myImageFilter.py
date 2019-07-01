# myImageFilter.py

#import math
import numpy as np
from scipy import signal


def imageFilter(img0,h):
    
    h_row,h_col = h.shape
    row,col = img0.shape
    
    h2 = np.resize(h,(row,col))
    
    H = np.fft.rfft2(h2)
    G = np.fft.rfft2(img0)
    
    Y = np.multiply(G,H)
    
    img1 = np.fft.irfft2(Y)   
    
    return img1



def myImageFilter(img0, h):

    #img1 = imageFilter(img0,h)
    
    img1 = signal.convolve2d(img0,h,mode='same')
    
    
    #h_row,h_col = h.shape
   # row_pad = h_row//2
    #col_pad = h_col//2
    #PAD
    #img_pad = np.pad(img0,((row_pad,row_pad),(col_pad,col_pad)),'constant')

    #row,col = img0.shape
    #img1 = np.zeros(img0.shape)
    
    #for i in range(row_pad,row+row_pad): #Start at pad value since 0,1... is pad
     #   for j in range(col_pad,col+col_pad): #python range is not inclusie of last value 
     #       boxed = img_pad[i-row_pad:i+row_pad+1,j-col_pad:j+col_pad+1]
     #       img1[i - row_pad,j-col_pad] = (h*boxed).sum()  #/h.sum()
            
    return img1
