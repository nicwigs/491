####################### Create Filter bank
import math
import numpy as np
from scipy import ndimage
def gaussian(shape=(3, 3), sigma=0.5):
    """
    2D gaussian filter
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
def laplacian_of_gaussian(shape=(3, 3), sigma=0.5):
    """
    2D laplacian of gaussian filter
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    tmp1 = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    tmp2 = 1 - (x * x + y * y) / (2. * sigma * sigma)
    tmp = tmp1 * tmp2
    h = -tmp / (np.pi * math.pow(sigma, 4))

    return h
def createFilterBank():

    # Code to generate reasonable filter bank
    gaussianScales = [1, 2, 4, 8, math.sqrt(2) * 8]
    logScales = [1, 2, 4, 8, math.sqrt(2) * 8]
    dxScales = [1, 2, 4, 8, math.sqrt(2) * 8]
    dyScales = [1, 2, 4, 8, math.sqrt(2) * 8]

    filterBank = []

    for scale in gaussianScales:
        siz = 2 * math.ceil(scale * 2.5) + 1
        tmp = gaussian(shape=(siz, siz), sigma=scale)
        filterBank.append(tmp)

    for scale in logScales:
        siz = 2 * math.ceil(scale * 2.5) + 1
        tmp = laplacian_of_gaussian(shape=(siz, siz), sigma=scale)
        filterBank.append(tmp)

    xgrad = np.array([[-1, 0, 1]])
    for scale in dxScales:
        siz = 2 * math.ceil(scale * 2.5) + 1
        f = gaussian(shape=(siz, siz), sigma=scale)
        f = ndimage.correlate(f, xgrad)
        filterBank.append(f)

    for scale in dyScales:
        f = gaussian(shape=(siz, siz), sigma=scale)
        f = ndimage.correlate(f, xgrad.transpose())
        filterBank.append(f)

    return filterBank
##############End Filter Bank
################Start extract responses
def RGB2Lab(R, G=None, B=None):

    if G is None and B is None:
        B = np.float64(R[:, :, 2])
        G = np.float64(R[:, :, 1])
        R = np.float64(R[:, :, 0])

    if np.max(R) > 1.0 or np.max(G) > 1.0 or np.max(B) > 1.0:
        R = R / 255.0
        G = G / 255.0
        B = B / 255.0

    M, N = R.shape
    s = M * N

    # Set a threshold
    T = 0.008856
    RGB = np.concatenate((np.reshape(R, (1, s)), np.reshape(G, (1, s)),
                         np.reshape(B, (1, s))), axis=0)
    MAT = np.array([[0.412453, 0.357580, 0.180423],
                   [0.212671, 0.715160, 0.072169],
                   [0.019334, 0.119193, 0.950227]]
                   )
    XYZ = np.matmul(MAT, RGB)

    X = XYZ[0, :] / 0.950456
    Y = XYZ[1, :]
    Z = XYZ[2, :] / 1.088754

    XT = X > T
    YT = Y > T
    ZT = Z > T

    fX = XT * np.power(X, 1.0 / 3.0) + (~XT) * (7.787 * X + 16.0 / 116.0)

    # Compute L
    Y3 = np.power(Y, 1.0 / 3.0)
    fY = YT * Y3 + (~YT) * (7.787 * Y + 16.0 / 116.0)
    L = YT * (116 * Y3 - 16.0) + (~YT) * (903.3 * Y)
    fZ = ZT * np.power(Z, 1.0 / 3.0) + (~ZT) * (7.787 * Z + 16 / 116)

    # Compute a and b
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    L = np.reshape(L, [M, N])
    a = np.reshape(a, [M, N])
    b = np.reshape(b, [M, N])

    L = np.stack([L, a, b], axis=2)

    return L
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
########### End response extraction 
######Start Random points
def getRandomPoints(I, alpha):    
    x,y,z = I.shape    
    x_rand = np.random.random_integers(0,x-1,size = (alpha,1))
    y_rand = np.random.random_integers(0,y-1,size = (alpha,1))
    points = np.concatenate((x_rand,y_rand),axis=1)    
    
    return points
####end random points
###Start Harris points
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
#### End harris points
    
import os
import pickle
import sklearn.cluster
from joblib import Parallel, delayed

def filterPoints(method,filters,alpha,K,source,name):
    fname1 = os.path.join(source,name)
    img = sp.ndimage.imread(fname1)
    if img.ndim == 3: #only use color images
            
        pixelResponses = np.zeros((alpha,3*len(filters)))        
        imageResponse = extractFilterResponses(img,filters) #apply filters      
        #Get alpha points for each image
        if method == 'Random':
            points = getRandomPoints(img,alpha)
        elif method == 'Harris':
            points = getHarrisPoints(img,alpha,K)    
        #Extract filter responses at points
        for row in range(0,alpha):
            x,y = points[row,:]
            pixelResponses[row] = imageResponse[x,y,:]
        return pixelResponses

#For windows machine, all code needs to be pretected by this 
    #to avoid recursive spawning
    #Also imports of other functions caused spawning, hence why they are above
if __name__ == '__main__':
    def getDictionary(imgPaths, alpha,K, method):
    
        ncores = 4
        data = pickle.load(open('../data/traintest.pkl','rb'))
        train = data['train_imagenames']
        filters = createFilterBank()  
        testset = train[0:100]
        
        print(('Starting a pool of workers with %d cores\n') % ncores)
        
        results = Parallel(n_jobs=ncores, verbose=11) \
        (delayed(filterPoints)(method,filters,alpha,K,imgPaths,name) for name in testset)
        #results = Parallel(n_jobs=ncores, verbose=11)(delayed(filterPoints)(i) for i in range(10))
        #result=Parallel(n_jobs=8)(delayed(math.sqrt)(i) for i in range(10))
        
        results = [i for i in results if isinstance(i, np.ndarray)] #gets rid of NONE from grey scale
        pixelResponses = np.vstack((results))
    
        d = sklearn.cluster.KMeans(n_clusters=K).fit(pixelResponses)
        dictionary = d.cluster_centers_
        
        fout = os.path.join(imgPaths,'dictionary' + method + '.npz')
        np.savez(fout,dictionary = dictionary, filterBank = filters)
            
        return dictionary

    alpha = 50      #Number of points desired 
    k = 0.04       #Should be 0.04-0.06 is good
    K = 100          #Number of clusters
    #testin Q1.3
    imgPaths = '../data/' 
        
    dictonary_Random = getDictionary(imgPaths, alpha,K, 'Random')