from skimage.feature import BRIEF,corner_fast,corner_peaks,match_descriptors,plot_matches
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
#returns points in x,y form (col,row)
def MatchPics(I1,I2):
    
    if I1.ndim == 3:
        I1 = rgb2gray(I1)   
    if I2.ndim == 3:
        I2 = rgb2gray(I2)
    
    points1 = corner_peaks(corner_fast(I1,n=12,threshold=0.15),min_distance=1)
    points2 = corner_peaks(corner_fast(I2,n=12,threshold=0.15),min_distance=1)
    
    extractor = BRIEF()
    
    extractor.extract(I1,points1)
    points1 = points1[extractor.mask]
    descriptors1 = extractor.descriptors
    
    extractor.extract(I2,points2)
    points2 = points2[extractor.mask]
    descriptors2 = extractor.descriptors
    
    matches = match_descriptors(descriptors1,descriptors2,metric = 'hamming',cross_check=True)
    
    #these points are y,x (row,col)
    locs1 = points1[matches[:,0]]
    locs2 = points2[matches[:,1]]
    #Change to x,y (col,row)
    xy1 = np.array([locs1[:,1],locs1[:,0]])
    xy1 = xy1.transpose()
    xy2 = np.array([locs2[:,1],locs2[:,0]])
    xy2 = xy2.transpose()
    fig, ax = plt.subplots()
    plot_matches(ax,I1,I2,points1,points2,matches,keypoints_color='r',only_matches=True)#,matches_color='y')
    
    return [xy1,xy2]