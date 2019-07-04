
import numpy as np

E = np.array([[1,2],[3,4]])
Kp = np.array([[7,8],[9,10]])
K = np.array([[5,4],[7,6]])

Ki = np.linalg.inv(K)
Kpi = np.linalg.inv(Kp)
KpiT = np.transpose(Kpi)
KpT = np.transpose(Kp)

F = np.matmul(E,Ki)
F = np.matmul(KpiT,F)

E1 = np.matmul(KpiT,np.matmul(F,K)) #wrong
E2 = np.matmul(KpT,np.matmul(F,K)) #corrrect!!!!!!!!!
