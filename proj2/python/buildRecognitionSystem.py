import os
import numpy as np
import pickle

from getImageFeatures import getImageFeatures

imgPaths = '../intermediate/' 

data = pickle.load(open('../data/traintest.pkl','rb'))
train_imagenames = data['train_imagenames']
train_labels = data['train_labels']
 
methods = ['Random', 'Harris']

for method in methods:
    dictionary = np.load('dictionary'+method+'.npz')
    dictionary = dictionary['dictionary']
    dict_sz = len(dictionary)
    
    trainFeatures = np.zeros((len(train_imagenames),dict_sz))
    trainLabels = np.zeros((len(train_imagenames),1))
    
    for i in range(0,len(train_imagenames)):    
        fname = os.path.join(imgPaths,method,os.path.splitext(train_imagenames[i])[0]+'.npz')   
        print('Image %d \n' %i)
        #if the image was greyscaled there will be no .npz file
        #so check to see if the file exists
        try:
            wordMap = np.load(fname)
        except IOError:
            continue #grey scale image, dont do anything 
        
        wordMap = wordMap['wordMap']

        trainFeatures[i] = getImageFeatures(wordMap,dict_sz)
        trainLabels[i] = train_labels[i]

    fout = os.path.join(imgPaths,'vision' + method + '.npz')
    np.savez(fout,trainFeatures=trainFeatures, trainLabels = trainLabels)