import os
import pickle
import numpy as np
import scipy.ndimage
import sklearn.cluster
from joblib import Parallel, delayed 
from extractFilterResponses import extractFilterResponses

def filterPoints(method,filters,alpha,K,source,name):
    print('Made it \n')
    fname1 = os.path.join(source,name)
    img = scipy.ndimage.imread(fname1)
    if img.ndim == 3: #only use color images
            
        pixelResponses = np.zeros((alpha,3*len(filters)))        
        imageResponse = extractFilterResponses(img,filters) #apply filters      
    #    #Get alpha points for each image
    #    if method == 'Random':
    #        points = getRandomPoints(img,alpha)
    #    elif method == 'Harris':
    #        points = getHarrisPoints(img,alpha,K)    
    #    #Extract filter responses at points
    #    for row in range(0,alpha):
    #        x,y = points[row,:]
    #        pixelResponses[row] = imageResponse[x,y,:]
    return pixelResponses

def work(arg):    
    #import time 
    #print("Function receives the arguments as a list:", arg)
    # Split the list to individual variables:
    #i, j = arg    
    # All this work function does is wait 1 second...
    #time.sleep(1)    
    # ... and prints a string containing the inputs:
    #print("%s_%s" % (i, j))
    return arg

if __name__ == '__main__':
    
    # List of arguments to pass to work():
    arg_instances = [(1, 1), (1, 2), (1, 3), (1, 4)]
    # Anything returned by work() can be stored:
    #, backend="threading"

   # results = Parallel(n_jobs=4, verbose=11)(map(delayed(work), arg_instances))
    
    #print(results)
    
    from createFilterBank import createFilterBank  
    #from extractFilterResponses import extractFilterResponses
    from getRandomPoints import getRandomPoints
    from getHarrisPoints import getHarrisPoints
  
    alpha = 50      #Number of points desired 
    k = 0.04       #Should be 0.04-0.06 is good
    K = 100          #Number of clusters
    method = 'Random'
    #testin Q1.3
    imgPaths = '../data/' 
    ncores = 4
    data = pickle.load(open('../data/traintest.pkl','rb'))
    train = data['train_imagenames']
    filters = createFilterBank()  
    testset = train[0:4]
    
    #whatname = __name__
    
    print(('Starting a pool of workers with %d cores\n') % ncores)
    
    results = Parallel(n_jobs=ncores, verbose=11)(delayed(filterPoints)(method,filters,alpha,K,imgPaths,name) for name in testset)
    #results = Parallel(n_jobs=ncores, verbose=11)(map(delayed(filterPoints),testset))  
    
    pixelResponses = np.vstack((results))

    d = sklearn.cluster.KMeans(n_clusters=K).fit(pixelResponses)
    dictionary = d.cluster_centers_
    
    fout = os.path.join(imgPaths,'dictionary' + method + '.npz')
    np.savez(fout,dictionary = dictionary, filterBank = filters)
        

