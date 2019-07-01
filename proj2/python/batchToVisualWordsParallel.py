###Start Filter Bank
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
####End Filter Bank
###RGB2Lab for extracting filter responses 
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
###End rgb2lab
###Get visual words depends on filter response - 
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
#####end of filter response 
###Start getVisualWords 
def getVisualWords(I,dictionary,filterBank):
    
    print("Getting Visual Words")
    imgResponse = extractFilterResponses(I,filterBank) #apply filters, (H,W,3n) size
    H,W,filt = imgResponse.shape
    imgResponse = imgResponse.reshape((H*W,filt)) #each row is a point
    
    dist = sp.spatial.distance.cdist(imgResponse,dictionary,metric='euclidean')
    
    wordMap = dist.argmin(axis=1) #find minimum of each row aka what cluster is closest, returns a shape (H*w,1)
    wordMap = wordMap.reshape((H,W)) #2D 
    
    return wordMap
####end getVisualWords 
import os
import pickle
import scipy as sp
from joblib import Parallel, delayed

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def processImage(source, destination, fname, filterBank, dictionary):
    fname1 = os.path.join(source, fname)
    fname2 = os.path.join(destination, os.path.splitext(fname)[0] + '.npz')
    img = sp.ndimage.imread(fname1)
    if img.ndim == 3: #we only want color images
      #  img = rgb2gray(img)

   # img = np.float64(img) / 255.0
        wordMap = getVisualWords(img, dictionary, filterBank)
        np.savez(fname2, wordMap=wordMap)

#For windows machine, all code needs to be pretected by this 
    #to avoid recursive spawning
    #Also imports of other functions caused spawning, hence why they are above
if __name__ == '__main__':

    # number of cores to use
    # set this value appropriately
    ncores = 4
    source = '../data/'
    methods = ['Random', 'Harris']
    
    # load the files and texton dictionary
    data = pickle.load(open('../data/traintest.pkl', 'rb'))
    mapping = data['mapping']
    all_imagenames = data['all_imagenames']
    nimages = len(all_imagenames)
    filterBank = createFilterBank()
    
    testset = all_imagenames[0:2]
    
    for method in methods:
        destination = os.path.join('../intermediate/', method)
        dictionary = np.load('dictionary' + method + '.npz')
        dictionary = dictionary['dictionary']
    
        if os.path.isdir(destination) is False:
                os.makedirs(destination)
    
        for category in mapping:
            if os.path.isdir(os.path.join(destination, category)) is False:
                os.makedirs(os.path.join(destination, category))
    
        print(('Starting a pool of workers with %d cores\n') % ncores)
        results = Parallel(n_jobs=ncores, verbose=11) \
        (delayed(processImage)(source, destination, name, filterBank, dictionary) for name in testset)
