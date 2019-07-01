import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from getImageDistance import getEuclideanDistance
from getImageDistance import getChiSquaredDistance
from getImageFeatures import getImageFeatures

source = '../intermediate/'
data = pickle.load(open('../data/traintest.pkl','rb'))
test = data['test_imagenames']
test_labels = data['test_labels']

#method = 'Harris'
methods = ['Random', 'Harris']
metrics = ['Eucldean','ChiSquared']

for method in methods:    #Harris or Random
    vision= np.load('../intermediate/vision' + method + '.npz')
    trainFeatures = vision['trainFeatures']
    trainLabels = vision['trainLabels']    
    train_cnt, cluster_cnt = trainFeatures.shape  
    
    for metric in metrics:      #Eucldean or ChiSquared    
        if metric == 'Eucldean':
            neighbors = KNeighborsClassifier(n_neighbors=1,metric=getEuclideanDistance)
        else:
            neighbors = KNeighborsClassifier(n_neighbors=1,metric=getChiSquaredDistance)
        trainLabels = np.ravel(trainLabels)
        neighbors.fit(trainFeatures,trainLabels)
        confusion = np.zeros((8,8))

        for i in range(0,len(test)):    
            fname = os.path.join(source,method,os.path.splitext(test[i])[0]+'.npz')
            try:
                wordMap = np.load(fname)
            except IOError:
                continue #grey scale image, dont do anything             
            wordMap = wordMap['wordMap']              
            features = getImageFeatures(wordMap,cluster_cnt)   
            
            actual = np.int(test_labels[i])
            classified = np.int(neighbors.predict(features)[0])            
            confusion[actual-1,classified-1] = confusion[actual-1,classified-1] + 1
            
        accuracy = np.trace(confusion)/np.sum(confusion) #correct/total         
        print("Using {} Dictionary and {} distance metric".format(method,metric))
        print("Accuracy = {}".format(accuracy))
        print(confusion)
        
        

    
    
    