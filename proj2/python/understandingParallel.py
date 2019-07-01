#from extractFilterResponses import extractFilterResponses


#from createFilterBank import createFilterBank
#from getRandomPoints import getRandomPoints
#from getHarrisPoints import getHarrisPoints
# A function that can be called to do work:
def work(arg):    
    import time 
    print("Function receives the arguments as a list:", arg)
    # Split the list to individual variables:
    i, j = arg    
    # All this work function does is wait 1 second...
    time.sleep(1)    
    # ... and prints a string containing the inputs:
    print("%s_%s" % (i, j))
    return "%s_%s" % (i, j)

if __name__ == '__main__':

    
    from joblib import Parallel, delayed 
    import os
    import pickle
    import numpy as np
    import scipy as sp
    import sklearn.cluster
    from extractFilterResponses import extractFilterResponses
    # List of arguments to pass to work():
    arg_instances = [(1, 1), (1, 2), (1, 3), (1, 4)]
    # Anything returned by work() can be stored:
    #, backend="threading"

    results = Parallel(n_jobs=4, verbose=11)(map(delayed(work), arg_instances))
    
    print(results)
