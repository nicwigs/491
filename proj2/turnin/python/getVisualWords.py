
import scipy as sp

from extractFilterResponses import extractFilterResponses

def getVisualWords(I,dictionary,filterBank):
    
    print("Getting Visual Words")
    imgResponse = extractFilterResponses(I,filterBank) #apply filters, (H,W,3n) size
    H,W,filt = imgResponse.shape
    imgResponse = imgResponse.reshape((H*W,filt)) #each row is a point
    
    dist = sp.spatial.distance.cdist(imgResponse,dictionary,metric='euclidean')
    
    wordMap = dist.argmin(axis=1) #find minimum of each row aka what cluster is closest, returns a shape (H*w,1)
    wordMap = wordMap.reshape((H,W)) #2D 
    
    return wordMap

#a = np.array([[[1,2,3],[12,13,14]],[[1,2,3],[12,13,14]]])
#b = np.array([[7,8,9],[10,11,12]])

#c = sp.spatial.distance.cdist(a,b,metric='euclidean')
#d = c.argmin(axis=1)
