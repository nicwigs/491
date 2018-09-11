# myHoughLines.py

import numpy as np
import copy

#import os
#import scipy as sp
#Looks left,right, up and down for how many pixels it takes until the pixel
#value is some percent of the maximum value. Then creates a square around the
#max value with dimensions nxn,n being the smallest distance to a pixel value
#of percent*maximum to implement non maximal suppression.
def mySuppression(H,row,col):
    percent = 0.05
    peak = H[row,col]
    row_sz,col_sz = H.shape
    
    right = 0
    left =0
    up = 0
    down = 0    
    while (col+right+1 <col_sz) and (H[row,col+right+1] > peak*percent):
        right = right + 1
    while (col-left-1 >= 0) and (H[row,col-left-1] > peak*percent):
        left = left + 1
    while (row-up-1 >= 0) and (H[row-up-1,col] > peak*percent):
        up = up + 1
    while (row+down+1 < row_sz) and (H[row+down+1,col] > peak*percent):
        down = down + 1
    
    row_shift = min(up,down)
    col_shift = min(right,left)
    
    #add 1 since closed range
    H[row-row_shift:row+row_shift+1,col-col_shift:col+col_shift+1] = 0
   # H[row-up:row+down+1,col-left:col+right+1] = 0
    
    #Used when printing out.....
    #resultsdir = '../results'
    #fname = os.path.join(resultsdir,  'maxSup.png')
    #sp.misc.imsave(fname, H / np.max(H))
 
    return H

def myHoughLines(H, nLines):
    
    H_suppress = copy.copy(H)
    
    rhos = np.zeros((nLines,1))
    thetas = np.zeros((nLines,1))
    
    for i in range(0,nLines):
        row,col = np.unravel_index(np.argmax(H_suppress, axis=None),H_suppress.shape)
        
        rhos[i] = row
        thetas[i] = col
        
        H_suppress = mySuppression(H_suppress,row,col)
    
    return rhos, thetas
