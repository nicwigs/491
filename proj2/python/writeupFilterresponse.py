
import os
from createFilterBank import createFilterBank
from extractFilterResponses import extractFilterResponses
import scipy as sp

fname = '../data/campus/sun_ahprylpgnmgqiyuz.jpg'

img = sp.ndimage.imread(fname)

#testing for Q1.1 Filter responses
filters = createFilterBank()
Im = extractFilterResponses(img,filters)    
image1 = Im[:,:,0:3] #guassian 0,3,6,9,12,|| 15,18,21,24,27 || ,30,33,36,39,42, || 45,48,51,54
image2 = Im[:,:,30:33]
image3 = Im[:,:,45:48]

sp.misc.imsave('../og.png', img)
sp.misc.imsave('../filter1.png', image1)
sp.misc.imsave('../filter2.png', image2)
sp.misc.imsave('../filter3.png', image3)