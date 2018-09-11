# batchToVisualWords.py

# Vishnu Naresh Boddeti | CSE-491 HW2
#
# Does parallel computation of the visual words
#
# Input:
#   ncores - number of cores to use (default 2)

import os
import pickle
import numpy as np
import scipy as sp
from joblib import Parallel, delayed
from getVisualWords import getVisualWords
from createFilterBank import createFilterBank


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def processImage(source, destination, fname):
    fname1 = os.path.join(source, fname)
    fname2 = os.path.join(destination, os.path.splitext(fname)[0] + '.npz')
    img = sp.ndimage.imread(fname1)
    if img.ndim == 3:
        img = rgb2gray(img)

    img = np.float64(img) / 255.0
    wordMap = getVisualWords(img, filterBank, dictionary)
    np.savez(fname2, wordMap=wordMap)


# number of cores to use
# set this value appropriately
ncores = 4
source = '../data/'
methods = ['Random', 'Harris']

# load the files and texton dictionary
data = pickle.load(open('../data/traintest.pkl', 'rb'))
mapping = data['mapping']
all_imagenames = data['all_imagenames']
nimages = len(all_imagenames)
filterBank = createFilterBank()

for method in methods:
    destination = os.path.join('../intermediate/', method)
    dictionary = np.load('dictionary' + method + '.npz')
    dictionary = dictionary['dictionary']

    if os.path.isdir(destination) is False:
            os.mkdir(destination)

    for category in mapping:
        if os.path.isdir(os.path.join(destination, category)) is False:
            os.mkdir(os.path.join(destination, category))

    print(('Starting a pool of workers with %d cores\n') % ncores)
    results = Parallel(n_jobs=ncores)(delayed(processImage)(source, destination, name) for name in all_imagenames)
