import numpy as np
from computeH_norm import computeH_norm

def correspondences(locs,randIndx):
    for i in range(4):
        temp = np.array([locs[randIndx[i],:]])
        if i == 0:
            points = temp
        else:
            points = np.concatenate((points,temp))
    return points

def my_ransac(locs1,locs2,threshold): #source, destination 
    
    randIndx = np.random.randint(0,locs1.shape[0],4)    #random 4 index
    x1 = correspondences(locs1,randIndx)                #Get the four correspondences
    x2 = correspondences(locs2,randIndx)
    
    H = computeH_norm(x1,x2)                            #H from my 4 random correspondences
    inliers = np.zeros((locs1.shape[0],1))              #initalize inliers
    
    for i in range(locs1.shape[0]):                     #Check all point correspondences 
        estimated = np.array([[locs1[i,0]],[locs1[i,1]],[1]]) #x is col, y is row hence 1 then 0 
        estimated = np.matmul(H,estimated)
        if estimated[2,0] == 0: #sometimes divide by 0 error
            estimated[2,0] = 0.000000001
        dist = np.sqrt((estimated[0,0]/estimated[2,0]-locs2[i,0])**2+(estimated[1,0]/estimated[2,0]-locs2[i,1])**2)
        inliers[i] = (dist<threshold)
    
    return [H,inliers]

def computeH_ransac(locs1,locs2,iterations,threshold):#source, destination 

    inliers = np.zeros((locs1.shape[0],1))          #initalize inliers
    cnt = 0                                         #Counts number of inliers
    for i in range(iterations):
        [newH,newInliers] = my_ransac(locs1,locs2,threshold)
        newCnt = newInliers.sum()
        if cnt < newCnt:
            inliers = newInliers
            bestH2to1 = newH
            cnt = inliers.sum()
            
    indx = inliers.nonzero()[0]                     #Indecies of the inliers 
    x1 = np.take(locs1,indx,axis=0)                 #points from image1 that are in inliers
    x2 = np.take(locs2,indx,axis=0)                 #points from image2 that are in inliers
    
    bestH2to1 = computeH_norm(x1,x2)                #Finally, compute H with the inliers
    
    return [bestH2to1, inliers]