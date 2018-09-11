import matplotlib.pyplot as plt
import scipy as sp

from getRandomPoints import getRandomPoints
from getHarrisPoints import getHarrisPoints

alpha = 500      #Number of points desired 
k = 0.04       #Should be 0.04-0.06 is good
K = 100          #Number of clusters

fname = '../data/campus/sun_abslhphpiejdjmpz.jpg'
img = sp.ndimage.imread(fname)

im = plt.imread(fname)
implot = plt.imshow(im)   


#testing Q1.2 random points
points = getRandomPoints(img,alpha)
row_points = points[:,0]
col_points = points[:,1]
plt.scatter(x = col_points,y=row_points, c='b',s = 5)
plt.show()

im = plt.imread(fname)
implot = plt.imshow(im)  
    
#Testing Q1.2 getHarrisPoints
points = getHarrisPoints(img,alpha,k)

row_points = points[:,0]
col_points = points[:,1]
plt.scatter(x = col_points,y=row_points, c='b',s = 5)
plt.show()