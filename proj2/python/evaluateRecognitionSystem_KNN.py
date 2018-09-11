import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from getImageDistance import getEuclideanDistance
from getImageDistance import getChiSquaredDistance
from getImageFeatures import getImageFeatures

source = '../intermediate/'
data = pickle.load(open('../data/traintest.pkl','rb'))
test = data['test_imagenames']
test_labels = data['test_labels']

method = 'Random'

vision= np.load('../intermediate/vision' + method + '.npz')
trainFeatures = vision['trainFeatures']
trainLabels = vision['trainLabels']    
train_cnt, cluster_cnt = trainFeatures.shape   

accuracy = np.zeros(41)
best_confusion = np.zeros((8,8))

for k in range(1,41):
    print('Current K: %d' % k)
    neighbors = KNeighborsClassifier(n_neighbors=k,metric=getChiSquaredDistance)  
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
        
    accuracy[k] = np.trace(confusion)/np.sum(confusion) #correct/total         
    if k == np.argmax(accuracy): #save best k value
        best_confusion = confusion
        
best_k = np.argmax(accuracy)
print("Best k number of nearest neighbors was {} with accuracy {}".format(best_k,accuracy[best_k]))
print(best_confusion)

plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.xlabel('K Nearest Neighbors')
plt.title('Accuracies')

        
