import numpy as np

def getEuclideanDistance(hist1, hist2):
    dist = (np.sum([((h1-h2)**2) for (h1,h2) in zip(hist1,hist2)]))**(1/2)
    return dist

def getChiSquaredDistance(hist1,hist2):
    safety = 1e-9
    dist = 0.5*np.sum([((h1-h2)**2)/(h1+h2+safety) for (h1,h2) in zip(hist1,hist2)])
    return dist

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[4,5,6],[7,8,9]])
c = [getEuclideanDistance(x,y) for (x,y) in zip(a,b)]

