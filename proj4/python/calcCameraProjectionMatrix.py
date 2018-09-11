import numpy as np
from camera2 import camera2
def calcCameraProjectionMatrix(K1,K2,E):
    #Assuming not rotation or translation in first image
    Rt1 = np.concatenate((np.identity(3),np.zeros((3,1))),axis=1)
    P1 = np.matmul(K1,Rt1)
    
    extrinsics = camera2(E)
    
    
    
    return [P1,P2]