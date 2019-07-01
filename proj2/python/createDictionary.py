from getDictionaryParallelScriptrev2 import getDictionary

alpha = 50      #Number of points desired 
k = 0.04       #Should be 0.04-0.06 is good
K = 100          #Number of clusters
#testin Q1.3
imgPaths = '../data/' 
    
dictonary_Random = getDictionary(imgPaths, alpha,K, 'Random')