import io
import scipy.io as matio
import os
import os.path
import numpy as np
from PIL import Image
import time
import re

import torch
import torch.utils.data
import torch.nn.parallel as para
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter

# —Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————
root_path = '/home/lily/Desktop/food/'  # /home/lily/Desktop/food/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
image_folder = 'ready_chinese_food'  # scaled_images ready_chinese_food
image_path = os.path.join(root_path, image_folder, '/')

file_path = os.path.join(root_path, 'SplitAndIngreLabel/')
ingredient_path = os.path.join(file_path, 'IngreLabel.txt')
glove_path = os.path.join(root_path, 'SplitAndIngreLabel/', 'glove.6B.300d.txt')

train_data_path = os.path.join(file_path, 'TR.txt')
validation_data_path = os.path.join(file_path, 'VAL.txt')
test_data_path = os.path.join(file_path, 'TE.txt')

result_path = root_path + 'results/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

test_path = root_path + 'test/'
if not os.path.exists(test_path):
    os.makedirs(test_path)

train_path = root_path + 'train/'
if not os.path.exists(train_path):
    os.makedirs(train_path)







#load whole test dataset
ingredient_all = matio.loadmat(file_path + 'ingredient_test_feature.mat')['ingredient_test_feature'][0:33154,:]
label_all = matio.loadmat(file_path + 'test_label.mat')['test_label'][0,:]

with io.open(file_path + 'TE.txt', encoding='utf-8') as file:
    path_to_images = file.read().split('\n')[:-1]

with open(file_path + 'IngredientList.txt', 'r', encoding="utf8") as f:
    ingre_list = f.read().split('\n')[:-1]

def get_ingre(ingre_vec):
    ingre_table = []

    for i in range(0,ingre_vec.shape[0]):
        index = np.where(ingre_vec[i,:] == 1)[0]

        ingre_words = []
        for i in index:
            ingre_words.append(ingre_list[i])

        ingre_table.append(ingre_words)
    return ingre_table



def get_img(index):
    for i in index:
        img_path = root_path + image_folder + path_to_images[i]
        img = Image.open(img_path).convert('RGB')
        img.save(test_path + 'img_'+str(i)+'.jpg')



#index = [3994,4305,10384,24365,31529]
index = [4365,4400,4468,4498,4528]
get_img(index)
ingre_words = get_ingre(ingredient_all[index,:])

print(ingre_words)
a=1



#trans = transforms.ToTensor()
#img = trans(Image.open(test_path+'0.jpg').convert('RGB'))

# #read sample
# ingredient_test = matio.loadmat(test_path + 'ingredients.mat')['ingredients']
# label_test = matio.loadmat(test_path + 'labels.mat')['labels'][0,:]
#
# sampleID = 0
# ingre_sample = ingredient_test[sampleID,:]
# label_sample = label_test[sampleID]
#
#
#
#
# sample_dis = matio.loadmat(test_path + 'sample_distance.mat')['sample_distance'][0,:]
# sample_labels = matio.loadmat(test_path + 'sample_label.mat')['sample_label'][0,:]
#
# index = np.where(sample_dis == 2)[0]
#
# table = []
# for x in index:
#     if int(sample_labels[x]) == label_sample:
#         table.append(x)
#
#
#


# #compute distance between sample and test dataset
# distance = np.zeros(ingredient_all.shape[0])
# label = np.zeros(ingredient_all.shape[0])
#
# for i in range(0,ingredient_all.shape[0]):
#     print('img {}'.format(i))
#     dis = np.sum(ingre_sample[:] * ingredient_all[i,:])
#     distance[i] = dis
#     label[i] = label_all[i]
#
#
# matio.savemat(test_path + 'sample_distance.mat', {'sample_distance': distance})
# matio.savemat(test_path + 'sample_label.mat', {'sample_label': label})
#
# a=1











# img = Image.open(test_path+'0.jpg').convert('RGB')
# image = np.array(img)
# R = image[:,:,0]
# G = image[:,:,1]
# B = image[:,:,2]
#
# imgR = Image.fromarray(R)
# imgG = Image.fromarray(G)
# imgB = Image.fromarray(B)
# imgR.save(test_path + 'DenoisedImage1.jpg')
# imgG.save(test_path + 'DenoisedImage2.jpg')
# imgB.save(test_path + 'DenoisedImage3.jpg')
#
# a=1