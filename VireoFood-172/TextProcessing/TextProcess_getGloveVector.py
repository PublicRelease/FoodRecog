import io
import scipy.io as matio
import os
import os.path
import numpy as np
import re

# —Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————
root_path = '/home/lily/Desktop/food/'  # /home/lily/Desktop/food/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
file_path = os.path.join(root_path, 'SplitAndIngreLabel/')
ori_path = file_path + 'origin_data/'
glove_path = os.path.join(root_path, 'SplitAndIngreLabel/','glove.6B.300d.txt')


#wordVector = matio.loadmat(file_path + 'wordVector.mat')['wordVector']


#load input matrix (data,ingredient/word)


# #generate glove vectors
# with io.open(glove_path, encoding='utf-8') as file:
#     lines = file.read().split('\n')[:-1]
#     if len(lines) != 400000:
#         print('error in slitting glove text!!!')
#
# num_lines = len(lines)
# num_dim = 300
#
# glove_head = []
# glove_vector = np.zeros((num_lines,num_dim))
#
#     #get words and vectors in glove
# i=0
# for line in lines:
#     print("processing {}-th line".format(i))
#     line = line.split()
#     glove_head.append(line[0])
#     glove_vector[i] = line[1:]
#
#     i+=1
#
# matio.savemat(file_path + 'glove_head.mat', {'glove_head': glove_head})
# matio.savemat(file_path + 'glove_vector.mat', {'glove_vector': glove_vector})


    #Load glove head vectors
glove_head = matio.loadmat(file_path + 'glove_head.mat')['glove_head']
glove_vector = matio.loadmat(file_path + 'glove_vector.mat')['glove_vector']

    #load word list of ingredient
wordList = matio.loadmat(ori_path + 'wordList.mat')['wordList']
num_word = wordList.shape[0]


    #match to extract vectors from
p=0 #indicate the index of words in wordList
wordVector = np.zeros((num_word,300))
count = 0
for word in wordList:
    print(p)
    q = 0
    for glove_word in glove_head:
        if re.match(word, glove_word):
            wordVector[p,:] = glove_vector[q,:]
            print('word {} matches glove word {}'.format(p,q))
            count+=1
            break
        q+=1
    p+=1

print(count)
matio.savemat(file_path + 'wordVector.mat', {'wordVector': wordVector})

a = 1