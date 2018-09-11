
import numpy as np
import scipy as sp

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def getHarrisPoints(I, alpha, k):
    
    if I.ndim == 3:
        I = rgb2gray(I)
    
    Ix = sp.ndimage.sobel(I,axis = 0, mode = 'constant')
    Iy = sp.ndimage.sobel(I,axis = 1, mode = 'constant')
    
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    sigma= 0.4
    Sxx = sp.ndimage.filters.gaussian_filter(Ixx,sigma)
    Syy = sp.ndimage.filters.gaussian_filter(Iyy,sigma)
    Sxy = sp.ndimage.filters.gaussian_filter(Ixy,sigma)
    
    Sxx_flat = Sxx.flatten()
    Syy_flat = Syy.flatten()
    Sxy_flat = Sxy.flatten()
    
    m = np.array([Sxx_flat,Sxy_flat,Sxy_flat,Syy_flat])
    M = np.reshape(m.T, (Sxx_flat.size,2,2),order='C')
    
    R = np.linalg.det(M)-k*(np.trace(M,axis1=1,axis2=2))**2
      
    flat_indices = R.argsort()[-alpha:][::-1]
    points = np.unravel_index(flat_indices,Sxx.shape) #returns tuple
    points = np.asarray(points) #make array
    points = points.T

    return points

