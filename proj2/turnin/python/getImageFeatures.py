
import numpy as np
import sklearn.preprocessing

def getImageFeatures(wordMap, dictionarySize):
    h = np.zeros(dictionarySize)
    
    cluster,count = np.unique(wordMap, return_counts = True)    #counts for each cluster
    h[cluster] = count                                          #replace clusters with correct counts
    h = np.reshape(h,(1,dictionarySize))                    #desired shape is 1xK 
    h = sklearn.preprocessing.normalize(h,norm='l1')            #Normalize since different image sizes
    return h
