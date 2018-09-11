import numpy as np

#Returns array of alpha points in image. 

def getRandomPoints(I, alpha):    
    x,y,z = I.shape    
    x_rand = np.random.random_integers(0,x-1,size = (alpha,1))
    y_rand = np.random.random_integers(0,y-1,size = (alpha,1))
    points = np.concatenate((x_rand,y_rand),axis=1)    
    
    return points
