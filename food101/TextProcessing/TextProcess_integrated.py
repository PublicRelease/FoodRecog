import io
import scipy.io as matio
import os.path
import numpy as np
import re

# —Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————

with open('ingredients_simplified.txt', 'r', encoding="utf8") as f:
    ingre_list = f.read().split('\n')[:-1]

wordList = [] #record the list of words in ingredients

ingre2word_map = np.zeros((len(ingre_list),300))
num_words = 0 #total counts for individual words
ingre2index_map = np.zeros((len(ingre_list),30))


for i in range(len(ingre_list)):
    temp_number = 0
    line = ingre_list[i]
    words = re.split(" |,|\?|\.",line)
    temp_wordList = []
    for word in words:
        if word in wordList:
            ingre2word_map[i,wordList.index(word)] = 1

        else:
            wordList.append(word)
            num_words+=1
            ingre2word_map[i,num_words-1] = 1
max_number =0
matio.savemat('wordList.mat', {'wordList': wordList})
ingre2word_map = ingre2word_map[:,0:num_words]
for i in range(len(ingre_list)):
    temp_number = 0
    line = ingre2word_map[i]
    #print(line)
    for j in range(len(line)):
        if line[j] ==1:
            ingre2index_map[i,temp_number] = j
            temp_number+=1
    if temp_number>max_number:
        max_number = temp_number


ingre2index_map = np.array(ingre2index_map)
matio.savemat('ingre2word_map.mat', {'ingre2word_map': ingre2word_map})
matio.savemat('ingre2index_map.mat',{'ingre2index_map':ingre2index_map})


def gene(name):
    label_array = []
    txt_name = name +'_label.txt'
    with open(txt_name) as file:
        lines = file.readlines()
        wordIndicator = []
        indexVector = []
        for label in lines:
            wordIndicator.append(ingre2word_map[int(label[:-1])])
            indexVector.append(ingre2index_map[int(label[:-1])])
    wordIndicator = np.array(wordIndicator)
    indexVector = np.array(indexVector)
    matio.savemat('wordIndicator_'+name+'.mat',{'wordIndicator_'+name:wordIndicator})
    matio.savemat('indexVector_'+name+'.mat',{'indexVector_'+name:indexVector})

gene('train')
gene('validation')
gene('test')

