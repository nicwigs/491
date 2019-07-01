import os
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from createFilterBank import createFilterBank
from extractFilterResponses import extractFilterResponses
from getRandomPoints import getRandomPoints
from getHarrisPoints import getHarrisPoints
from getDictionary import getDictionary
from getVisualWords import getVisualWords
from getImageFeatures import getImageFeatures

datadir = '../data/airport'
resultsdir = '..'

imglist = []
for file in os.listdir(datadir):
    if file.endswith(".jpg"):
        imglist.append(file)

imglist.sort()
for i in range(len(imglist)):
       
    # read in images
    fname = os.path.join(datadir, imglist[i])
    img = sp.ndimage.imread(fname)

    im = plt.imread(fname)
    implot = plt.imshow(im)   
    
    #testing Q1.2 random points
    points = getRandomPoints(img,5)
   # x_points = points[:,0]
   # y_points = points[:,1]
   # plt.scatter(x = x_points,y=y_points, c='r',s = 20)
    #plt.show()
    
    alpha = 5      #Number of points desired 
    k = 0.04       #Should be 0.04-0.06 is good
    K = 3           #Number of clusters
    #testin Q1.3
    imgPaths = '../data/'    
    dictonary = getDictionary(imgPaths, alpha,K, 'Harris')
    filters = createFilterBank()    

    #testing Q2.1
    wordMap = getVisualWords(img,dictonary,filters)
    
    #testing Q2.2
    h = getImageFeatures(wordMap,K)
    
    
    #Testing Q1.2 getHarrisPoints
    points = getHarrisPoints(img,alpha,k)

    x_points = points[:,0]
    y_points = points[:,1]
    plt.scatter(x = x_points,y=y_points, c='b',s = 10)
    plt.show()

    #testing for Q1.1 Filter responses
    filters = createFilterBank()
    Im = extractFilterResponses(img,filters)    
    image1 = Im[:,:,0:3]
    


    # everything below here just saves the outputs to files
    filename_w_ext = os.path.basename(imglist[i])
    filename, file_extension = os.path.splitext(filename_w_ext)
    fname = os.path.join(resultsdir, filename + '_01edge.png')
    #temp = np.uint8(255.0 * np.sqrt(Im / np.max(Im)))
    temp = image1
    
    sp.misc.imsave(fname, temp)

