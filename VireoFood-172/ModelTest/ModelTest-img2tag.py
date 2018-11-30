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
root_path = '/mnt/FoodRecog/'  # /home/lily/Desktop/food/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
image_folder = 'ready_chinese_food'  # scaled_images ready_chinese_food
image_path = os.path.join(root_path, image_folder)

file_path = os.path.join(root_path, 'SplitAndIngreLabel/')
ingredient_path = os.path.join(file_path, 'IngreLabel.txt')

train_data_path = os.path.join(file_path, 'TR.txt')
validation_data_path = os.path.join(file_path, 'VAL.txt')
test_data_path = os.path.join(file_path, 'TE.txt')

result_path = root_path + 'results_lstm3finetune/'
if not os.path.exists(result_path):
    os.makedirs(result_path)


# —Create dataset———————————————————————————————————————————————————————————————————————————————————————————————————————
def default_loader(path):
    img_path = image_path + path

    jpgfile = Image.open(img_path).convert('RGB')

    return jpgfile


class FoodData(torch.utils.data.Dataset):
    def __init__(self, train_data=False, test_data=False, transform=None,
                 loader=default_loader):

        # load image paths / label file
        if train_data:
            {}

        elif test_data:
            with io.open(test_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')

            ingredients = matio.loadmat(file_path + 'wordIndicator_test.mat')['wordIndicator_test']

        ingredients = ingredients.astype(np.float32)

        self.path_to_images = path_to_images
        self.ingredients = ingredients
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        # get ingredients 353-D vector
        ingredient = self.ingredients[index, :]

        return img, ingredient

    def __len__(self):
        return len(self.path_to_images)


# —Model———————————————————————————————————————————————————————————————————————————————————————————————————————

# Encoder network for image
__all__ = [
    'vgg16_bn',
    'vgg19_bn',
]

model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False

    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict
                           and not re.match(k, 'classifier.0.weight')
                           and not re.match(k, 'classifier.6.weight')
                           and not re.match(k, 'classifier.6.bias')
                           }

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False

    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict
                           and not re.match(k, 'classifier.0.weight')
                           and not re.match(k, 'classifier.6.weight')
                           and not re.match(k, 'classifier.6.bias')
                           }

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


# encoder network for textual channel
class encoder_t(nn.Module):
    def __init__(self, max_seq=30, num_hidden=4096, num_key_ingre=5):
        super(encoder_t, self).__init__()

        # load glove vectors
        wordVector = matio.loadmat(file_path + 'wordVector.mat')['wordVector']
        # add a zero vector on top for padding_idx
        wordVector = np.concatenate([np.zeros((1, wordVector.shape[1])), wordVector], 0)

        self.embedding = nn.Embedding(wordVector.shape[0], wordVector.shape[1], padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(wordVector))

        self.num_hidden = num_hidden
        self.num_key_ingre = num_key_ingre

        # self.register_parameter('h0_en', None)
        self.gru = nn.GRU(wordVector.shape[1], self.num_hidden)  # nn.GRU(wordVector.shape[1], num_hidden, dropout=0.1)

        # linear layers for gru attention
        self.ws1 = nn.Linear(num_hidden, num_hidden)
        self.ws2 = nn.Linear(num_hidden, num_key_ingre)
        self.ingre2att = nn.Linear(num_key_ingre, 1)
        self.relu = nn.LeakyReLU()

        self._initialize_weights()

    # def init_hidden(self, input):
    #    self.h0_en = nn.Parameter(input.new(input.size()).normal_(0, 1))
    def forward(self, y):
        # compute latent vectors
        # indexVector, num_words_per_data, word_label = getIndexVector(y, self.max_seq)
        # indexVector = torch.from_numpy(indexVector).long().cuda(3)
        encoder_t_embeds = self.embedding(y)
        encoder_t_embeds = encoder_t_embeds.permute(1, 0, 2)

        # obtain gru output of hidden vectors
        # if self.h0_en is None:
        #    self.init_hidden(torch.zeros((1, y.shape[0], self.num_hidden), requires_grad=True))
        h0_en = Parameter(torch.zeros((1, y.shape[0], self.num_hidden), requires_grad=True))
        self.gru.flatten_parameters()
        y_embeds, _ = self.gru(encoder_t_embeds, h0_en.cuda(set_gpu_others))

        att_y_embeds, multi_attention = self.getAttention(y_embeds)

        return att_y_embeds, encoder_t_embeds, multi_attention

    def getAttention(self, y_embeds):
        # y_embeds = self.dropout(y_embeds)  # (seq, batch, hidden)
        y_embeds = y_embeds.transpose(0, 1)  # (batch, seq, hidden)
        # compute multi-focus self attention by a two-layer mapping
        # (batch, seq, hidden) -> (batch, seq, hidden) -> (batch, seq, self.num_key_ingre)
        multi_attention = self.ws2(self.relu(self.ws1(y_embeds)))
        # compute attended embeddings in terms of focus
        multi_attention = multi_attention.transpose(1, 2)  # (batch, self.num_key_ingre, seq)
        att_y_embeds = multi_attention.bmm(y_embeds)  # (batch, self.num_key_ingre, hidden)
        # compute the aggregated hidden vector
        att_y_embeds = self.ingre2att(att_y_embeds.transpose(1, 2)).squeeze(2)  # (batch, hidden)
        return att_y_embeds, multi_attention.transpose(1, 2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


# decoder network for textual channel
class decoder_t(nn.Module):
    def __init__(self, num_hidden=4096, num_glove=300, num_seq=30, num_word=309):
        super(decoder_t, self).__init__()

        # load glove vectors
        wordVector = matio.loadmat(file_path + 'wordVector.mat')['wordVector']
        # add a zero vector on top for padding_idx
        wordVector = np.concatenate([np.zeros((1, wordVector.shape[1])), wordVector], 0)
        self.embedding = nn.Embedding(wordVector.shape[0], wordVector.shape[1], padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(wordVector))

        self.num_glove = num_glove
        self.seq = num_seq
        self.num_word = num_word
        self.softmax = nn.Softmax()
        self.relu = nn.LeakyReLU()

        # self.register_parameter('h0_de', None)
        self.hiddenMap1 = nn.Linear(num_hidden + num_glove, num_hidden)
        self.hiddenMap2 = nn.Linear(num_hidden, num_hidden)

        self.gru = nn.GRU(num_hidden, num_glove)  # nn.GRU(num_hidden, num_glove, dropout=0.1)

        self.wordpredict1 = nn.Linear(num_glove, self.num_glove)
        self.wordpredict2 = nn.Linear(num_glove, self.num_word)

        self.relu = nn.LeakyReLU()
        self._initialize_weights()

    # def init_hidden(self, input):
    #    self.h0_de = nn.Parameter(input.new(input.size()).normal_(0, 1))
    def forward(self, y):
        # Use latent_y = (batch, num_hidden) as input to predict a sequence of ingredient words
        # y has size (batch,num_hidden)

        h_de = []  # store the output hidden vectors of gru
        gru_predicts = []  # store the predicts of gru for words

        h0_de = Parameter(torch.zeros((1, y.shape[0], self.num_glove), requires_grad=True))
        # if self.h0_en is None:
        #    self.init_hidden(torch.zeros((1, y.shape[0], self.num_glove), requires_grad=True))
        current_input = torch.cat([y, torch.zeros(y.shape[0], self.num_glove).cuda(set_gpu_others)], 1).unsqueeze(
            0)  # (1, batch, num_hidden+num_glove)
        current_input = self.hiddenMap2(self.relu(self.hiddenMap1(current_input)))
        # print('current_input: {}'.format(current_input.shape))
        prev_hidden = h0_de.cuda(set_gpu_others)
        # print('prev_hidden: {}'.format(prev_hidden.shape))

        for i in range(0, self.seq):  # for each of the max_seq for decoder
            # NOTE: current_hidden = prev_hidden, we use different notations to clarify their roles
            current_hidden, prev_hidden = self.gruLoop(current_input, prev_hidden)
            # save gru output
            h_de.append(current_hidden)
            # compute next input to gru, the glove embedding vector of the current predicted word
            current_input, wordPredicts = self.getNextInput(y, current_hidden)

            gru_predicts.append(wordPredicts)

        return torch.cat(gru_predicts, 0), torch.cat(h_de, 0)  # make it a tensor (seq, batch, num_word)

    def getNextInput(self, y, current_hidden):
        # get embedding of the predicted words
        wordPredicts = self.wordpredict2(self.relu(self.wordpredict1(current_hidden))).squeeze(
            0)  # (1, batch, num_glove) -> (batch, num_word)
        wordIndex = torch.argmax(wordPredicts, dim=1)  # (batch, 1)
        embeddings = self.embedding(
            wordIndex + 1)  # (batch,num_glove) note that the index 0 of Embedding is for non-entry
        # fuse the embedding with y_latent using a non-linear mapping to extract sufficient information for the next word
        next_input = self.hiddenMap2(self.relu(self.hiddenMap1(torch.cat([y, embeddings], 1)))).unsqueeze(0)

        return next_input, wordPredicts.unsqueeze(0)

    def gruLoop(self, current_input, prev_hidden):
        # use it to avoid a modification of prev_hidden with inplace operation
        output, hidden = self.gru(current_input, prev_hidden)
        return output, hidden

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


set_gpu_encoder = 1
set_gpu_others = 0


# entire model
class MyModel(nn.Module):
    def __init__(self, num_key_ingre=5, max_seq=30):
        super(MyModel, self).__init__()
        # network for image channel
        self.encoder = vgg19_bn().cuda(set_gpu_encoder)
        self.vgg_map2vec = nn.Linear(512 * ((image_size[0] // (2 ** 5)) ** 2), 4096).cuda(set_gpu_others)
        self.vgg_linear = nn.Linear(4096, 4096).cuda(set_gpu_others)

        # text channel modules
        self.encoder_t = encoder_t().cuda(set_gpu_others)
        self.decoder_t = decoder_t().cuda(set_gpu_others)

        # classifier
        # self.classifier_v = nn.Linear(blk_len, 172).cuda(set_gpu_others)
        # self.classifier_t = nn.Linear(blk_len, 172).cuda(set_gpu_others)

        # domain transfer
        # self.trans_img2l = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)
        # self.trans_text2l = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)

        # cross channel generation
        self.cross_img2z = nn.Linear(4096, 4096).cuda(set_gpu_others)
        self.cross_z2text = nn.Linear(4096, 4096).cuda(set_gpu_others)

        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()

        self._initialize_weights()

    def forward(self, x):  # x:image, y:ingredient
        # compute image latent vectors & recons
        x_latent_maps = self.encoder(x)
        x_latent = self.get_latent(x_latent_maps.cuda(set_gpu_others))

        # map x_latent to y_latent
        y_latent_recon = self.get_y_recon(x_latent)

        # word prediction using y_latent_recon
        gru_predicts_recon, decoder_t_embeds_recon = self.decoder_t(y_latent_recon)  # (seq, batch, words)

        return gru_predicts_recon.transpose(0, 1).cuda(set_gpu_others)

    def get_y_recon(self, x_latent):
        y_latent_recon = self.cross_z2text(
            self.dropout(self.relu(self.cross_img2z(
                x_latent))))

        return y_latent_recon

    def get_latent(self, x_latent_maps):
        x_latent = x_latent_maps.view(x_latent_maps.size(0), -1)
        x_latent = self.dropout(self.relu(self.vgg_map2vec(x_latent)))
        x_latent = self.dropout(self.relu(self.vgg_linear(x_latent)))
        return x_latent

    def get_predicts_with_align(self, x_latent, y_latent):
        # compute features in the transferred latent domain
        # image channel
        x_latent2l = self.trans_img2l(x_latent)
        predicts = self.classifier_v(x_latent2l)
        # text channel
        y_latent2l = self.trans_text2l(y_latent)
        predicts_t = self.classifier_t(y_latent2l)

        return predicts, predicts_t

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


# —Manual settings———————————————————————————————————————————————————————————————————————————————————————————————————————
# Image Info
no_of_channels = 3
image_size = [256, 256]  # [64,64]

# changed configuration to this instead of argparse for easier interaction
CUDA = 1  # 1 for True; 0 for False
SEED = 1
BATCH_SIZE = 32
LOG_INTERVAL = 10
blk_len = 1536

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}

# Download or load dataset
# shuffle data at every epoch

test_loader = torch.utils.data.DataLoader(
    FoodData(train_data=False, test_data=True,
             transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)


# ------------------------------------------------------------------
def top_match(wordPredicts, labels):
    wordList = matio.loadmat(file_path + 'wordList.mat')['wordList']
    PredictIndex = np.zeros(
        30)  # record the index of the top-1 predicted words in wordList (of length 10), at instance level
    precisionCount = np.zeros(30)  # count the hit of correct predicts in top 1-30, at instance level
    TopCounts = np.zeros(10)  # record the number of correct predicts in top 1-10, at word level
    total_TopCounts = np.zeros(10)  # record the number of correct predicts in top 1-10, at word level, for the batch
    total_precision_count = np.zeros(
        30)  # proportion of groundtruth words that are in the top-n predicted words.  /((np.arange(30)+1)*labels.shape[0]) average is done outside the function for compute overall performance
    wordCount = 0  # total number of words
    hitCount = 0  # total number of hits in top-1 prediction
    total_avg_recall = np.zeros(
        30)  # proportion of groundtruth words that are retrieved in the top-n predicted words.  / labels.shape[0] average is done outside the function for compute overall performance

    for i in range(0, wordPredicts.shape[0]):  # for each batch item

        if np.sum(labels[i, :]) == 0:  # to detect possible testing data with zero entries
            print('no input at data {}'.format(i))
            continue

        TopCounts[:] = 0
        PredictIndex[:] = 0
        precisionCount[:] = 0
        index = np.where(labels[i, :] == 1)[0]  # get the indexes of non-zero entries e.g. [12,67]
        wordCount += index.shape[0]

        with io.open(result_path + 'img2tag.txt', 'a', encoding='utf-8') as file:
            file.write('True Tags: ')
            for p in range(0, index.shape[0]):
                if p == 0:
                    file.write('{}'.format(wordList[index[0]].split()[0]))
                else:
                    file.write(' {}'.format(wordList[index[p]].split()[0]))

            file.write('\n')
            file.write('Rank in Predicts: ')

        for j in range(0, index.shape[0]):  # for each seq item
            # print('index:{}'.format(index.shape))
            # print('indexj:{}'.format(index[j]))
            # print('indexj:{}'.format(j))
            # print('wordPredicts:{}'.format(wordPredicts[i, j, :].shape))
            sorted_predicts = wordPredicts[i, j,
                              :].argsort()  # .cpu().data.numpy().argsort() ranked list of indexes, value:low -> high
            PredictIndex[j] = sorted_predicts[-1]
            position_in_list = np.where(sorted_predicts == index[j])[0][0]
            rank = labels.shape[1] - 1 - position_in_list

            if rank == 0:
                hitCount += 1

            if rank < 10:
                TopCounts[rank:10] += 1
            with io.open(result_path + 'img2tag.txt', 'a', encoding='utf-8') as file:
                if j == 0:
                    file.write('{} {},'.format(rank, wordList[sorted_predicts[position_in_list]].split()[0]))
                elif j == index.shape[0] - 1:
                    file.write(' {} {}'.format(rank, wordList[sorted_predicts[position_in_list]].split()[0]))
                else:
                    file.write(' {} {},'.format(rank, wordList[sorted_predicts[position_in_list]].split()[0]))

        total_TopCounts += TopCounts

        with io.open(result_path + 'img2tag.txt', 'a', encoding='utf-8') as file:
            file.write('\n')
            file.write('avgProbability in word Top-n prediction: ')
            for p in range(0, 10):
                if p == 0:
                    file.write('{},'.format(TopCounts[p] / index.shape[0]))
                elif p == 9:
                    file.write(' {}'.format(TopCounts[p] / index.shape[0]))
                else:
                    file.write(' {},'.format(TopCounts[p] / index.shape[0]))

        for j in range(index.shape[0], 30):  # for each seq
            sorted_predicts = wordPredicts[i, j,
                              :].argsort()  # .cpu().data.numpy() ranked list of indexes, value:low -> high
            PredictIndex[j] = sorted_predicts[-1]  # compute all predicted words

            # print('\n')
        # print('Processing batch item: {}'.format(i))
        # print('Groundtruth indexes of words: {}:'.format(index))
        # print('Predicted indexes of words: {}:'.format(PredictIndex))
        for p in range(0, index.shape[0]):
            #    print('Processing seq {}'.format(p))
            position_in_list = np.where(PredictIndex == index[p])[0]
            #    print('Current word index is: {}'.format(index[p]))
            #    print('Location in predicted words is: {}'.format(position_in_list))

            if position_in_list.shape[0] != 0:
                precisionCount[int(position_in_list[0]):labels.shape[
                    1]] += 1  # If the word appears multiple times, count the location of the first time
        avg_precision = precisionCount[:] / (np.arange(30) + 1)
        avg_recall = precisionCount[:] / index.shape[0]  # recall for this batch item

        total_precision_count += precisionCount[:]
        total_avg_recall += avg_recall[:]

        with io.open(result_path + 'img2tag.txt', 'a', encoding='utf-8') as file:
            file.write('\n')
            file.write('avg_precision for item: ')
            for p in range(0, 30):
                if p == 0:
                    file.write('{},'.format(avg_precision[p]))
                elif p == 30 - 1:
                    file.write(' {}'.format(avg_precision[p]))
                else:
                    file.write(' {},'.format(avg_precision[p]))

            file.write('\n')
            file.write('avg_recall for item: ')
            for p in range(0, 30):
                if p == 0:
                    file.write('{},'.format(avg_recall[p]))
                elif p == 30 - 1:
                    file.write(' {}'.format(avg_recall[p]))
                else:
                    file.write(' {},'.format(avg_recall[p]))

            file.write('\n')
            file.write('Total predicted words: ')
            for p in range(0, 30):
                if p == 0:
                    file.write('{},'.format(wordList[int(PredictIndex[p])].split()[0]))
                elif p == 30 - 1:
                    file.write(' {}'.format(wordList[int(PredictIndex[p])].split()[0]))
                else:
                    file.write(' {},'.format(wordList[int(PredictIndex[p])].split()[0]))
            file.write('\n')
            file.write('\n')

    avg_precision_word = total_TopCounts  # probability that a word can be found in top-n predicts. /wordCount average is done outside the function for compute overall performance

    return total_precision_count, total_avg_recall, avg_precision_word, wordCount


def test():
    # toggle model to test / inference mode
    print('testing starts..')
    model.eval()
    avg_precision_total = np.zeros(30)
    avg_recall_total = np.zeros(30)
    avg_precision_word_total = np.zeros(10)
    wordCount_total = 0

    total_time = time.time()

    # each data is of BATCH_SIZE (default 128) samples
    for test_batch_idx, (data, wordIndicator_test) in enumerate(test_loader
                                                                ):
        # --------------------------------------------------------------------------------------------------------------------------------
        # for effective code debugging
        # if test_batch_idx == 1:
        #    break
        # print('batch %',batch_idx)
        # ---------------------------------------------------------------------------------------------------------------------------------
        start_time = time.time()

        if CUDA:
            # make sure this lives on the GPU
            data = data.cuda(set_gpu_encoder)

        # predicts_V, x = model(data)
        wordPredicts = model(data)

        # compute accuracy
        wordPredicts = wordPredicts.cpu().data.numpy()
        wordIndicator_test = wordIndicator_test.data.numpy()

        avg_precision, avg_recall, avg_precision_word, wordCount = top_match(wordPredicts, wordIndicator_test)

        avg_precision_total += avg_precision[:]
        avg_recall_total += avg_recall[:]
        avg_precision_word_total += avg_precision_word[:]
        wordCount_total += wordCount

        # top 1-30 performance at this batch
        avg_precision_batch = avg_precision[:] / ((np.arange(30) + 1) * data.shape[0])
        avg_recall_batch = avg_recall[:] / data.shape[0]
        avg_precision_word_batch = avg_precision_word[:] / wordCount

        print(
            'Testing batch: {} | avg_precision_batch:{} | avg_recall_batch:{} | avg_precision_word_batch:{} | Time:{} | Total_Time:{}'.format(
                test_batch_idx, avg_precision_batch[0], avg_recall_batch[0], avg_precision_word_batch[0],
                round((time.time() - start_time), 4),
                round((time.time() - total_time), 4)))

    print('total_average_recall is {}'.format(avg_recall_total))
    print('number of data is {}'.format(len(test_loader.dataset)))
    print('avg recall is {}'.format(avg_recall_total / len(test_loader.dataset)))

    print(
        '====> Test set: \n avg_precision_total:{} | \n avg_recall_total:{} | \n avg_precision_word_total:{} | \n Total Time:{}'.format(
            avg_precision_total / ((np.arange(30) + 1) * len(test_loader.dataset)),
            avg_recall_total / len(test_loader.dataset), avg_precision_word_total / wordCount_total,
            round((time.time() - total_time), 4)))

    return avg_precision_total / ((np.arange(30) + 1) * len(test_loader.dataset)), avg_recall_total / len(
        test_loader.dataset), avg_precision_word_total / wordCount_total


# —Model testing———————————————————————————————————————————————————————————————————————————————————————————————————————



def get_updateModel(path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    # temp = {k for k, v in pretrained_dict.items()}
    # with io.open(result_path + 'dict.txt', 'w', encoding='utf-8') as file:
    #    for item in temp:
    #        file.write('{}\n'.format(item))

    # for k, v in pretrained_dict.items():
    #    if k.startswith('cross_'):
    #        print(k)

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k.startswith('encoder.')
                       or k.startswith('vgg_')
                       or k.startswith('encoder_t.')
                       or k.startswith('decoder_t.')
                       or k.startswith('cross_')

                       }
    # k.startswith('trans_img2l.')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


# Model
model = MyModel()

# if CUDA:
#    model = nn.DataParallel(model).cuda()

max_index = 0
max_pre = 0
max_recall = 0
max_wordPre = 0

start = 13
end = 13

for i in range(start, end + 1):
    path = result_path + 'model' + str(i) + '.pt'
    model = get_updateModel(path)

    avg_precision, avg_recall, avg_precision_word = test()

    with io.open(result_path + 'img2tag_performance.txt', 'a', encoding='utf-8') as file:
        file.write(
            '\n ====> Test set:\n avg_precision_total:{}\n avg_recall_total:{}\n avg_precision_word_total:{}'.format(
                avg_precision, avg_recall, avg_precision_word))





