import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import skimage.color
import scipy.ndimage 


fname1 = '../data/desert/sun_adpbjcrpyetqykvt.jpg'
fname2 = '../intermediate/Random/desert/sun_adpbjcrpyetqykvt.npz'
fname3 = '../intermediate/Harris/desert/sun_adpbjcrpyetqykvt.npz'

img = sp.ndimage.imread(fname1)

im = plt.imread(fname1)
implot = plt.imshow(im) 
plt.show()  

wordMap = np.load(fname2)
wordMap = wordMap['wordMap']
im = skimage.color.label2rgb(wordMap)
implot = plt.imshow(im)
plt.show()

wordMap = np.load(fname3)
wordMap = wordMap['wordMap']
im = skimage.color.label2rgb(wordMap)
implot = plt.imshow(im)
plt.show()


