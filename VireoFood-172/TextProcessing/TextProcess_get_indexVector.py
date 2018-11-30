import io
import scipy.io as matio
import os.path
import numpy as np
import time
import torch

# —Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————
root_path = '/home/lily/Desktop/food/'  # /home/lily/Desktop/food/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
#file_path = os.path.join(root_path, 'SplitAndIngreLabel/origin_data/')
file_path = os.path.join(root_path, 'SplitAndIngreLabel/')



#load input matrix (data,ingredient)
wordIndicator_test = matio.loadmat(file_path + 'wordIndicator_val.mat')['wordIndicator_val']
#wordIndicator_test = matio.loadmat(file_path + 'wordIndicator_test.mat')['wordIndicator_test']
max_seq = 30 #The maximum number of words

indexVector_test = np.zeros((wordIndicator_test.shape[0],max_seq)) #tore the indexes of words (in our 309 word table) for each food item

for i in range(0,wordIndicator_test.shape[0]): #for each food item
    print('processing image ' + str(i))
    #get the index vector
    a = wordIndicator_test[i,:]
    index = np.where(a==1)[0]
    #assign the index to our indexVector
    indexVector_test[i,0:index.shape[0]] = index[:]+1


#matio.savemat(file_path + 'indexVector_val2.mat', {'indexVector_val2': indexVector_test})





















