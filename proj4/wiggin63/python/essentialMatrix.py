# essentialMatrix.py
import numpy as np

def essentialMatrix(F,K1,K2):
    E = np.matmul(np.transpose(K2),np.matmul(F,K1))
    return E
