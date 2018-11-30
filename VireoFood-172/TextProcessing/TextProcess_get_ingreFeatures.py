import io
import scipy.io as matio
import os
import os.path
import numpy as np
import re

root_path = '/home/lily/Desktop/food/'  # /home/lily/Desktop/food/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
image_path = os.path.join(root_path, 'ready_chinese_food/')
file_path = os.path.join(root_path, 'SplitAndIngreLabel/')
ingredient_path = os.path.join(file_path, 'IngreLabel.txt')

train_data_path = os.path.join(file_path, 'TR.txt')
validation_data_path = os.path.join(file_path, 'VAL.txt')
test_data_path = os.path.join(file_path, 'TE.txt')

#create ingredient train/validation/test features
with io.open(ingredient_path, encoding='utf-8') as file:
    ingrelabel = file.read().split('\n')[:-1]

num_all = len(ingrelabel)

index_map = np.arange(num_all) #limit the search space in the list of all lines of data



#process training data
with io.open(train_data_path, encoding='utf-8') as file:
    lines = file.read().split('\n')[:-1]

num_img = len(lines)
ingredient_train_feature = np.zeros((num_img,353))


i = 0
for line in lines:
    print('processing train line ' + str(i))
    for j in range(len(index_map)):
        head=ingrelabel[index_map[j]].split()[0]
        if re.match(line, head):
            ingredient_train_feature[i,:] = ingrelabel[index_map[j]].split()[1:]
            index_map = np.delete(index_map, j)
            break
    i+=1

ingredient_train_feature[np.where(ingredient_train_feature < 0)] = 0
matio.savemat(file_path + 'ingredient_train_feature.mat', {'ingredient_train_feature': ingredient_train_feature})
del ingredient_train_feature






#process test data
with io.open(test_data_path, encoding='utf-8') as file:
    lines = file.read().split('\n')[:-1]

num_img = len(lines)
ingredient_test_feature = np.zeros((num_img,353))


i = 0
for line in lines:
    print('processing test line ' + str(i))
    for j in range(len(index_map)):
        head=ingrelabel[index_map[j]].split()[0]
        if re.match(line, head):
            ingredient_test_feature[i,:] = ingrelabel[index_map[j]].split()[1:]
            index_map = np.delete(index_map, j)
            break
    i+=1


ingredient_test_feature[np.where(ingredient_test_feature < 0)] = 0
matio.savemat(file_path + 'ingredient_test_feature.mat', {'ingredient_test_feature': ingredient_test_feature})
del ingredient_test_feature



#process val data
with io.open(validation_data_path, encoding='utf-8') as file:
    lines = file.read().split('\n')[:-1]

num_img = len(lines)
ingredient_val_feature = np.zeros((num_img,353))


i = 0
for line in lines:
    print('processing val line ' + str(i))
    for j in range(len(index_map)):
        head=ingrelabel[index_map[j]].split()[0]
        if re.match(line, head):
            ingredient_val_feature[i,:] = ingrelabel[index_map[j]].split()[1:]
            index_map = np.delete(index_map,j)
            break
    i+=1


ingredient_val_feature[np.where(ingredient_val_feature < 0)] = 0
matio.savemat(file_path + 'ingredient_val_feature.mat', {'ingredient_val_feature': ingredient_val_feature})


