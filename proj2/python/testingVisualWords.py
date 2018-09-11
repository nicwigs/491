import os
import scipy as sp
import numpy as np
import pickle
from getVisualWords import getVisualWords
from createFilterBank import createFilterBank

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

ncores = 4
source = '../data/'
method = 'Harris'

# load the files and texton dictionary
data = pickle.load(open('../data/traintest.pkl', 'rb'))
mapping = data['mapping']
all_imagenames = data['all_imagenames']
nimages = len(all_imagenames)
filterz = createFilterBank()

dictionary = np.load('dictionary' + method + '.npz')
dictionary = dictionary['dictionary']

fname = all_imagenames[0]

fname1 = os.path.join(source, fname)
#fname2 = os.path.join(destination, os.path.splitext(fname)[0] + '.npz')
img = sp.ndimage.imread(fname1)
#if img.ndim == 3:
#    img = rgb2gray(img)

#img = np.float64(img) / 255.0
wordMap = getVisualWords(img, dictionary, filterz)
done = 1
#np.savez(fname2, wordMap=wordMap)