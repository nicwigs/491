import numpy as np

from RGB2Lab import RGB2Lab
from scipy import signal 

def extractFilterResponses(I, filterBank):

    L = RGB2Lab(I[:,:,0],I[:,:,1],I[:,:,2])
    
    x,y,z = L.shape
    z = z*len(filterBank)
    
    filterResponses = np.zeros((x,y,z))
    
    for filt_indx in range(len(filterBank)):
        for channel in range(0,3,1):
            colorChannel = L[:,:,channel]  
            filt = filterBank[filt_indx]
            filterResponses[:,:,filt_indx*3+channel] = signal.convolve2d(colorChannel,filt,mode='same')
            
        #print("Filter Number: {}".format(filt_indx))

    return filterResponses

