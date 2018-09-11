import numpy as np
def scale(points,M):
    xt = points/M
    T = np.array([[1/M,0,0],\
                [0,1/M,0],\
                [0,0,1]])
    return [xt,T]
def computeF(x1,x2):
    x = x1[:,0]
    y = x1[:,1]
    xp = x2[:,0]
    yp = x2[:,1]

    sz = np.size(x1,0)                              #Create Matrix A
    for i in range(sz):
        Ai = np.array([[x[i]*xp[i],x[i]*yp[i],x[i],xp[i]*y[i],y[i]*yp[i],y[i],xp[i],yp[i],1]])
        if i == 0:
            A = Ai
        else:
            A = np.concatenate((A,Ai),axis=0)
                                                    #Compute SVD
    U,S,Vt = np.linalg.svd(A,full_matrices=False)   #V is really V transpose 
    eigVec = Vt[Vt.shape[0]-1,:]                    #Get row since this is Vtranspose 
    F = eigVec.reshape((3,3))
                                                    #compute SVD of F to see rank
    U2,S2,Vt2 = np.linalg.svd(F,full_matrices=False)#V is really V transpose 
    S2[S2.size-1] = 0                               #Force smallest singular value to 0 to enforce rank 2
    S2 = np.diag(S2)
    temp = np.matmul(S2,Vt2)                        #now just multiply back out
    Fp = np.matmul(U2,temp)                         #DO NOT SCALE F LIKE WE DID H 
    return Fp
#x1, and x2 are Nx2 matrix each (x,y) coordinates  (x = col, y = row)
def eightpoint(x1,x2,M):
    [xt1,T1] = scale(x1,M)
    [xt2,T2] = scale(x2,M)

    F = computeF(xt1,xt2)
    F2to1 = np.matmul(F,T1)
    F2to1 = np.matmul(T2.transpose(),F2to1)         #Unlike H, its transform not inverse,DONT SCALE
    return F2to1