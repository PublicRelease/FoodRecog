import io
import scipy.io as matio
import os.path
import numpy as np


# —Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————
root_path = '/home/lily/Desktop/food/'  # /home/lily/Desktop/food/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
file_path = os.path.join(root_path, 'SplitAndIngreLabel/origin_data/')
te = os.path.join(root_path, 'SplitAndIngreLabel/')


#ingre2word_map = matio.loadmat(file_path + 'ingre2word_map.mat')['ingre2word_map']

with open(file_path + 'IngredientList.txt', 'r', encoding="utf8") as f:
    ingre_list = f.read().split('\n')[:-1]


wordList = [] #record the list of words in ingredients
ingre2word_map = np.zeros((len(ingre_list),1000))
num_words = 0 #total counts for individual words

for i in range(0,len(ingre_list)):
    print('process word {}'.format(i))
    words = ingre_list[i].split() #individual words in a gredient

    for word in words:
        if word in wordList:
            ingre2word_map[i,wordList.index(word)] = 1
        else:
            wordList.append(word)
            num_words += 1
            ingre2word_map[i,num_words-1] = 1

#matio.savemat(file_path + 'wordList.mat', {'wordList': wordList})



ingre2word_map = ingre2word_map[:,0:num_words]
#matio.savemat(file_path + 'ingre2word_map.mat', {'ingre2word_map': ingre2word_map})



