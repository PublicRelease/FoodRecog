import io
import scipy.io as matio
import os.path
import numpy as np
import time
import torch

# —Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————
root_path = '/home/lily/Desktop/food/'  # /home/lily/Desktop/food/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
file_path = os.path.join(root_path, 'SplitAndIngreLabel/origin_data/')
te = os.path.join(root_path, 'SplitAndIngreLabel/')



#load input matrix (data,ingredient)
ingredient_train_feature = matio.loadmat(te + 'ingredient_train_feature.mat')['ingredient_train_feature']

ingredient_val_feature = matio.loadmat(file_path + 'ingredient_val_feature.mat')['ingredient_val_feature']
ingredient_test_feature = matio.loadmat(file_path + 'ingredient_test_feature.mat')['ingredient_test_feature']


#load mapping from ingredient to words (ingredient,words)
ingre2word_map = matio.loadmat(file_path + 'ingre2word_map.mat')['ingre2word_map']



#compute the final word indicator for data
wordIndicator_train = ingredient_train_feature @ ingre2word_map
wordIndicator_val = ingredient_val_feature @ ingre2word_map
wordIndicator_test = ingredient_test_feature @ ingre2word_map

wordIndicator_train[np.where(wordIndicator_train>1)] = 1
wordIndicator_val[np.where(wordIndicator_val>1)] = 1
wordIndicator_test[np.where(wordIndicator_test>1)] = 1


matio.savemat(file_path + 'wordIndicator_train.mat', {'wordIndicator_train': wordIndicator_train})
matio.savemat(file_path + 'wordIndicator_val.mat', {'wordIndicator_val': wordIndicator_val})
matio.savemat(file_path + 'wordIndicator_test.mat', {'wordIndicator_test': wordIndicator_test})

















