
import math
import os
import pickle
import numpy as np
import scipy as sp
import sklearn.cluster
from joblib import Parallel, delayed

from extractFilterResponses import extractFilterResponses
from createFilterBank import createFilterBank
from getRandomPoints import getRandomPoints
from getHarrisPoints import getHarrisPoints

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
    
if __name__ == '__main__':
    def getDictionary(imgPaths, alpha,K, method):
    
        ncores = 4
        data = pickle.load(open('../data/traintest.pkl','rb'))
        train = data['train_imagenames']
        filters = createFilterBank()  
        testset = train[0:2]
        
        #whatname = __name__
        
        print(('Starting a pool of workers with %d cores\n') % ncores)
        
        results = Parallel(n_jobs=ncores, verbose=11)(delayed(filterPoints)(i) for i in range(10))
        #result=Parallel(n_jobs=8)(delayed(math.sqrt)(i) for i in range(10))
        
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