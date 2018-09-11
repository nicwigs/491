# myHoughTransform.py

import math
import numpy as np


def myHoughTransform(img, threshold, rhoRes, thetaRes):

    row_sz,col_sz = img.shape
    #row0 = row_sz/2
    #col0 = col_sz/2
    
    #rhoMax = np.sqrt(np.square(row0)+np.square(col0))
    rhoMax = np.sqrt(np.square(row_sz)+np.square(col_sz))
    
    rhoScale = np.arange(0,rhoMax+rhoRes,rhoRes)
    thetaScale = np.arange(0,math.pi+thetaRes,thetaRes)
    
    H = np.zeros((rhoScale.shape[0],thetaScale.shape[0]))
    
    
    
    for row in range(0,row_sz):
        for col in range(0,col_sz):
            if img[row,col] > threshold:                
                for theta_index in range(0,thetaScale.shape[0]):                    


                    rho = (col)*math.cos(thetaScale[theta_index])+ \
                    (row)*math.sin(thetaScale[theta_index])

                    if rho >= 0:       #Need positive rho                 
                        rho_index = (np.abs(rhoScale-rho)).argmin()
                        H[rho_index,theta_index] = H[rho_index,theta_index] +1
                    
    return H, rhoScale, thetaScale
