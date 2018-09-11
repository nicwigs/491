# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:26:27 2018

@author: nic
"""

import matplotlib.pyplot as plt
import math
import numpy as np

x = 10
y = 10

thetaRes = math.pi/16
theta = np.arange(0,math.pi+thetaRes,thetaRes)

rho = x*np.cos(theta)+y*np.sin(theta)

plt.plot(theta,rho)
plt.ylabel('Rho')
plt.xlabel('Theta')
plt.grid()
plt.show()
