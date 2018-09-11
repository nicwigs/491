# triangulate.py
import numpy as np

def triangulate(P1,x1,P2,x2):
    
    p1 = P1[0,:]        #array p1
    p2 = P1[1,:]
    p3 = P1[2,:]
    p1p = P2[0,:]        #array p1 from P2 (p1 prime)
    p2p = P2[1,:]
    p3p = P2[2,:]
    
    p1T = p1.transpose()
    p2T = p2.transpose()
    p3T = p3.transpose()
    p1pT = p1p.transpose()
    p2pT = p2p.transpose()
    p3pT = p3p.transpose()
    
    sz = x1.shape[0]
    for i in range(sz):
        x = x1[i,0]
        y = x1[i,1]
        xp = x2[i,0]
        yp = x2[i,1]
        
        A = np.array([y*p3T-p2T,
                      p1T-x*p3T,
                      yp*p3pT-p2pT,
                      p1pT-xp*p3pT])
    
        U,S,Vt = np.linalg.svd(A)   #V is really V transpose 
        eigVec = Vt[Vt.shape[0]-1,:]                    #Get row since this is Vtranspose 
        Xi = eigVec.reshape((1,4))                      
        Xi = Xi/Xi[0,3]                                 #unscale Xi = [x y z 1]
        #Xi = np.array([Xi[0,0:3]])                              #make Xi = [x y z ]
        if i == 0:
            X = Xi
        else:
            X = np.concatenate((X,Xi),axis=0)
    return X
