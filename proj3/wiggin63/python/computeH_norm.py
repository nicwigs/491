import numpy as np
from computeH import computeH

def translation(points):
    #point is in x,y (col,row)!
    x = points[:,0]
    y = points[:,1]
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    #translate the mean of the points to the origin
    xt = x - mean_x
    yt = y - mean_y
    #Calculate euclidian dist
    dist = np.sqrt(xt**2+yt**2)
    #Find max distance to origin
    indx = np.argmax(dist)
    max1 = dist[indx]
    #Max distance should be root(2)
    scale = np.sqrt(2)/max1
    #Apply Scale
    xt = xt*scale
    yt = yt*scale
    #Place back togther
    t = np.concatenate((np.transpose(np.array([xt])),np.transpose(np.array([yt]))),axis=1)
    
    return [scale,mean_x,mean_y,t]

def computeH_norm(x1,x2):
    #source, destination 
    [scale1,mean_x1,mean_y1,xt1] = translation(x1)
    T1 = np.array([[scale1,0,scale1*(-mean_x1)], \
                    [0,scale1,scale1*(-mean_y1)], \
                    [0,0,1]])
    
    [scale2,mean_x2,mean_y2,xt2] = translation(x2)
    T2 = np.array([[scale2,0,scale2*(-mean_x2)], \
                    [0,scale2,scale2*(-mean_y2)], \
                    [0,0,1]])
    
    H = computeH(xt1,xt2)
    H2to1 = np.matmul(H,T1)
    H2to1 = np.matmul(np.linalg.inv(T2),H2to1) #H = t2inverse*H*T1inverse
    H2to1 = H2to1 / H2to1[2, 2]
    
    return H2to1

