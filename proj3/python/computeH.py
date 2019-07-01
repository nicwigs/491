import numpy as np

def computeH(x1,x2):
    #source, destination 
    #go from x to xprime
    
    x = x1[:,0]
    y = x1[:,1]
    
    xp = x2[:,0]
    yp = x2[:,1]

    sz = np.size(x1,0)
    #Create Matrix A
    for i in range(sz):
        Ai = np.array([[-x[i],-y[i],-1,0,0,0,x[i]*xp[i],y[i]*xp[i],xp[i]],\
                       [0,0,0,-x[i],-y[i],-1,x[i]*yp[i],y[i]*yp[i],yp[i]]])
        if i == 0:
            A = Ai
        else:
            A = np.concatenate((A,Ai),axis=0)
        
    #Compute SVD
    U,S,V = np.linalg.svd(A) #V is really V transpose 
    #print(S)
    eigVal = S[S.size-1]
    eigVec = V[V.shape[0]-1,:]
    
    H2to1 = eigVec.reshape((3,3))
    H2to1 = H2to1 / H2to1[2, 2]
    
    return H2to1
