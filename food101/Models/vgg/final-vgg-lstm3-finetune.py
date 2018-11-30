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
root_path = './food-101/'  # /home/lily/Desktop/food/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
image_folder = 'images/'  # scaled_images ready_chinese_food
image_path = os.path.join(root_path, image_folder, '/')

file_path = ''
ingredient_path = ''
glove_path = os.path.join(root_path, 'SplitAndIngreLabel/', 'glove.6B.300d.txt')
#
# train_data_path = os.path.join(file_path, 'train.txt')
# validation_data_path = os.path.join(file_path, 'validation.txt')
# test_data_path = os.path.join(file_path, 'test.txt')

train_data_path = 'train.txt'
validation_data_path = 'validation.txt'
test_data_path = 'test.txt'

result_path = root_path + 'results2/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

test_path = root_path + 'test/'
if not os.path.exists(test_path):
    os.makedirs(test_path)

train_path = root_path + 'train/'
if not os.path.exists(train_path):
    os.makedirs(train_path)

# —Create dataset———————————————————————————————————————————————————————————————————————————————————————————————————————
def default_loader(path):
    img_path = root_path + image_folder + path
    jpgfile = Image.open(img_path).convert('RGB')
    return jpgfile


class FoodData(torch.utils.data.Dataset):
    def __init__(self, train_data=False, test_data=False, transform=None,
                 loader=default_loader):

        # load image paths / label file
        if train_data:
            with io.open(train_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            labels = matio.loadmat(file_path + 'train_label.mat')['train_label']
            ingredients = matio.loadmat(file_path + 'wordIndicator_train.mat')['wordIndicator_train']

            with io.open(validation_data_path, encoding='utf-8') as file:
                path_to_images1 = file.read().split('\n')
            labels1 = matio.loadmat(file_path + 'validation_label.mat')['validation_label']
            ingredients1 = matio.loadmat(file_path + 'wordIndicator_val.mat')[
                'wordIndicator_val']

            path_to_images = path_to_images + path_to_images1
            labels = np.concatenate([labels, labels1], 1)[0, :]
            ingredients = np.concatenate([ingredients, ingredients1], 0)

            indexVector = matio.loadmat(file_path + 'indexVector_train.mat')['indexVector_train']

        ingredients = ingredients.astype(np.float32)
        indexVector = indexVector.astype(np.long)
        self.path_to_images = path_to_images
        self.labels = labels
        self.ingredients = ingredients
        self.indexVector = indexVector
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        # get label
        label = self.labels[index]
        # get ingredients 353-D vector
        ingredient = self.ingredients[index, :]

        # get index vector of (batch, 30) for rnn embedding
        indexVector = self.indexVector[index, :]

        return img, label, ingredient, indexVector

    def __len__(self):
        return self.labels.shape[0]


# —Manual settings———————————————————————————————————————————————————————————————————————————————————————————————————————
# Image Info
no_of_channels = 3
image_size = [256, 256]  # [64,64]

# changed configuration to this instead of argparse for easier interaction
CUDA = 1  # 1 for True; 0 for False
SEED = 1
BATCH_SIZE = 32
LOG_INTERVAL = 10
learning_rate = 1e-4
blk_len = int(4096 * 3 / 8)  # 1536

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}

# Download or load dataset
# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(
    FoodData(train_data=True, test_data=False,
             transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

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
        self.classifier = nn.Linear(blk_len, 172).cuda(set_gpu_others)
        # self.classifier_t = nn.Linear(blk_len, 172).cuda(set_gpu_others)

        # domain transfer
        self.trans_img2l = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)
        self.trans_text2l = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)

        # cross channel generation
        self.cross_img2z = nn.Linear(4096, 4096).cuda(set_gpu_others)
        self.cross_z2text = nn.Linear(4096, 4096).cuda(set_gpu_others)

        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()

        self._initialize_weights()

    def forward(self, x, y):  # x:image, y:ingredient
        # compute image latent vectors & recons
        x_latent_maps = self.encoder(x)
        x_latent = self.get_latent(x_latent_maps.cuda(set_gpu_others))

        # map x_latent to y_latent
        y_latent_recon = self.get_y_recon(x_latent)

        # compute ingredient vectors
        att_y_latent, encoder_t_embeds, multi_attention = self.encoder_t(y)

        # compute v t predicts in domain adapted space
        predicts, predicts_t, AE = self.get_predicts_with_align(x_latent[:, 0:blk_len], att_y_latent[:, 0:blk_len])

        # word prediction using att_y_latent
        gru_predicts, decoder_t_embeds = self.decoder_t(att_y_latent)  # (seq, batch, words)

        # word prediction using y_latent_recon
        gru_predicts_recon, decoder_t_embeds_recon = self.decoder_t(y_latent_recon)  # (seq, batch, words)

        return predicts.cuda(set_gpu_others), predicts_t.cuda(set_gpu_others), gru_predicts.cuda(
            set_gpu_others), gru_predicts_recon.cuda(set_gpu_others), y_latent_recon.cuda(
            set_gpu_others), att_y_latent.cuda(set_gpu_others)

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
        predicts = self.classifier(x_latent2l)
        # text channel
        y_latent2l = self.trans_text2l(y_latent)
        predicts_t = self.classifier(y_latent2l)

        kl_loss = nn.KLDivLoss()
        AE = kl_loss(self.log_softmax(x_latent2l),
                     self.softmax(y_latent).detach())

        return predicts, predicts_t, AE

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


# —Model training & testing———————————————————————————————————————————————————————————————————————————————————————————————————————
def get_updateModel():
    model_dict = model.state_dict()

    # update image channel
    pretrained_dict = torch.load(file_path + 'finalModel_encoder.pt', map_location='cpu')

    extracted_dict_img = {k: v for k, v in pretrained_dict.items() if
                          k.startswith('encoder.')
                          or k.startswith('vgg_')
                          or k.startswith('classifier_v.')
                          or k.startswith('trans_img2l.')
                          }

    # update lstm channel
    pretrained_dict = torch.load(file_path + 'finalModel_lstm2.pt', map_location='cpu')
    extracted_dict_lstm = {k: v for k, v in pretrained_dict.items() if
                           k.startswith('encoder_t.')
                           or k.startswith('decoder_t.')
                           or k.startswith('classifier_t.')
                           or k.startswith('trans_text2l.')
                           }

    # pretrained_dict = torch.load(past_result_path + 'final-vgg-lstm3-finetune1/model13.pt', map_location='cpu')
    # extracted_dict_decoder = {k: v for k, v in pretrained_dict.items() if k.startswith('decoder_t.embedding')}

    model_dict.update(extracted_dict_img)
    model_dict.update(extracted_dict_lstm)
    # model_dict.update(extracted_dict_decoder)

    #    for k, v in pretrained_dict.items():
    #        if k.startswith('encoder_t.embedding'):
    #            model_dict['de'+k[2:]] = v



    model.load_state_dict(model_dict)

    return model


# Model
model = MyModel()
model = get_updateModel()


# pretrained_dict = torch.load(file_path + 'model9-words.pt')
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and (n.startswith('encoder') or #n.startswith('decoder'))}

# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    frozen_params = [p for n, p in model.named_parameters() if
                     not n.startswith('cross_')]  # and not n.startswith('decoder_t.')]
    # finetune_params = [p for n, p in model.named_parameters() if n.startswith('decoder_t.')]
    non_frozen_params = [p for n, p in model.named_parameters() if n.startswith('cross_')]
    params = [{'params': frozen_params, 'lr': 0},  # {'params': finetune_params, 'lr': 1e-6},
              {'params': non_frozen_params}]
    # params = [{'params': non_frozen_params}]

    # for param in frozen_params:
    #    param.requires_grad = False

    optimizer = optim.Adam(params, lr=lr)  # weight_decay=1e-3, lr=lr)
    return optimizer


optimizer = get_optim(learning_rate)

# optim.Adam(model.parameters(), weight_decay=1e-3,
#                      lr=learning_rate)  # .module.parameters(), lr=learning_rate)

# ------------------------------------------------------------------


# Loss
criterion = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss()
softmax = nn.Softmax()
logsoftmax = nn.LogSoftmax()


def getLSTMloss(gru_predicts, gru_predicts_recon, wordIndicator):
    lstm_loss = 0
    recon_lstm_loss = 0
    # embed_match_loss = 0
    gru_predicts = gru_predicts.transpose(0, 1)  # (seq, batch, num_word) -> (batch, seq, num_word)
    gru_predicts_recon = gru_predicts_recon.transpose(0, 1)
    # encoder_t_embeds = encoder_t_embeds.transpose(0, 1)
    # decoder_t_embeds = decoder_t_embeds.transpose(0, 1)
    wordIndicator = wordIndicator.numpy()  # word indicators
    total_num = 0

    for i in range(0, gru_predicts.shape[0]):  # for each batch item
        index = np.where(wordIndicator[i, :] == 1)[0]  # compute the index of non-zero entries
        for j in range(0, index.shape[0]):  # for each seq item
            total_num += 1
            lstm_loss += criterion(gru_predicts[i, j, :].unsqueeze(0),
                                   torch.from_numpy(np.array(index[j])).unsqueeze(0).cuda(set_gpu_others))
            recon_lstm_loss += criterion(gru_predicts_recon[i, j, :].unsqueeze(0),
                                         torch.from_numpy(np.array(index[j])).unsqueeze(0).cuda(set_gpu_others))

            # embed_match_loss += torch.sum((encoder_t_embeds[i, j, :] - decoder_t_embeds[i, j, :]) ** 2)

    return lstm_loss / total_num, recon_lstm_loss / total_num  # , embed_match_loss / total_num


def loss_function(predicts, predicts_t, gru_predicts, gru_predicts_recon, y_latent_recon, att_y_latent, ingredients,
                  labels):
    # image channel loss
    CE_V = criterion(predicts, labels - 1) * 1
    CE_T = criterion(predicts_t, labels - 1) * 1

    # ingredient channel loss
    lstm_loss, recon_lstm_loss = getLSTMloss(gru_predicts, gru_predicts_recon, ingredients)

    y_latent_transfer_loss = torch.sum((y_latent_recon - att_y_latent) ** 2) / y_latent_recon.shape[1] / labels.shape[
        0]  # kl_loss(logsoftmax(y_latent_recon), softmax(att_y_latent).detach()) #* 1e4

    # partial heterogeneous transfer
    # AE = AE  # * (1e5)  # torch.sum((x_latent - y_latent) ** 2) * (1e-3)

    # constraints on the attention weights of gru encode
    # Identity = torch.eye(multi_attention.shape[1]).unsqueeze(0)  # (1,seq,seq)
    # Identity = Identity.repeat(multi_attention.shape[0], 1, 1).cuda(set_gpu_others)  # (batch,seq,seq)
    # ones = torch.ones(multi_attention.shape[1], multi_attention.shape[1])
    # ones = ones.repeat(multi_attention.shape[0], 1, 1).cuda(set_gpu_others)  # (batch,seq,seq)

    # multi_attention_Transpose = torch.transpose(multi_attention, 1, 2).contiguous().cuda(set_gpu_others)  # (batch, num_key_ingredient, seq)
    # ATT = torch.sum((multi_attention.bmm(multi_attention_Transpose) * (ones - Identity)) ** 2) * (1e3)

    return CE_V, CE_T, lstm_loss, recon_lstm_loss, y_latent_transfer_loss  # , embed_match_loss, ATT


# ------------------------------------------------------------------

def top_match(predicts, labels):
    sorted_predicts = predicts.cpu().data.numpy().argsort()
    top1_labels = sorted_predicts[:, -1:][:, 0]
    match = float(sum(top1_labels == (labels - 1)))

    top5_labels = sorted_predicts[:, -5:]
    hit = 0
    for i in range(0, labels.size(0)):
        hit += (labels[i] - 1) in top5_labels[i, :]

    return match, hit


def train(epoch):
    # toggle model to train mode
    print('Training starts..')
    model.train()
    train_loss = 0
    top1_accuracy_total_V = 0
    top5_accuracy_total_V = 0
    total_time = time.time()

    for batch_idx, (data, labels, ingredients, indexVector) in enumerate(train_loader):
        # ---------------------------------------------------------------------------------------------------------------------------------
        # for effective code debugging
        # if batch_idx < len(train_loader)-20:
        #    print('skip batch {}'.format(batch_idx))
        #    continue
        # print('batch %',batch_idx)
        # ---------------------------------------------------------------------------------------------------------------------------------

        start_time = time.time()
        data = Variable(data)
        indexVector = Variable(indexVector)
        if CUDA:
            data = data.cuda(set_gpu_encoder)
            labels = labels.cuda(set_gpu_others)
            indexVector = indexVector.cuda(set_gpu_others)

        # obtain output from model
        predicts, predicts_t, gru_predicts, gru_predicts_recon, y_latent_recon, att_y_latent = model(data, indexVector)

        # loss
        CE_V, CE_T, lstm_loss, recon_lstm_loss, y_latent_transfer_loss = loss_function(predicts, predicts_t,
                                                                                       gru_predicts, gru_predicts_recon,
                                                                                       y_latent_recon, att_y_latent,
                                                                                       ingredients, labels)

        # optim for myModel with generator
        optimizer.zero_grad()
        loss = recon_lstm_loss + y_latent_transfer_loss  # lstm_loss  #embed_match_loss + ATT  # + CE_T + AE
        loss.backward()
        train_loss += loss.data
        optimizer.step()

        # compute accuracy
        predicts = predicts.cpu()
        labels = labels.cpu()

        matches_V, hits_V = top_match(predicts, labels)
        # top 1 accuracy
        top1_accuracy_total_V += matches_V
        top1_accuracy_cur_V = matches_V / float(labels.size(0))

        # top 5 accuracy
        top5_accuracy_total_V += hits_V
        top5_accuracy_cur_V = hits_V / float(labels.size(0))

        if epoch == 1 and batch_idx == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | lstm_loss: {:.4f} | recon_lstm_loss: {:.4f} | y_latent_recon_loss: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * labels.shape[0], len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.data,
                    CE_V.data, CE_T.data, lstm_loss.data, recon_lstm_loss.data, y_latent_transfer_loss.data,
                    top1_accuracy_cur_V, top5_accuracy_cur_V,
                    round((time.time() - start_time), 4),
                    round((time.time() - total_time), 4)))

            with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
                # print('write in-batch loss at epoch {} | batch {}'.format(epoch,batch_idx))
                file.write('%f\n' % (train_loss))

        elif batch_idx % LOG_INTERVAL == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | lstm_loss: {:.4f} | recon_lstm_loss: {:.4f} | y_latent_recon_loss: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * labels.shape[0], len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.data,
                    CE_V.data, CE_T.data, lstm_loss.data, recon_lstm_loss.data, y_latent_transfer_loss.data,
                    top1_accuracy_cur_V, top5_accuracy_cur_V,
                           round((time.time() - start_time), 4) * LOG_INTERVAL,
                    round((time.time() - total_time), 4)))

        # records current progress for tracking purpose
        with io.open(result_path + 'model_batch_train_loss.txt', 'w', encoding='utf-8') as file:
            file.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | lstm_loss: {:.4f} | recon_lstm_loss: {:.4f} | y_latent_recon_loss: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * labels.shape[0], len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.data,
                    CE_V.data, CE_T.data, lstm_loss.data, recon_lstm_loss.data, y_latent_transfer_loss.data,
                    top1_accuracy_cur_V, top5_accuracy_cur_V,
                           round((time.time() - start_time), 4) * LOG_INTERVAL,
                    round((time.time() - total_time), 4)))

    print(
        '====> Epoch: {} | Average loss: {:.4f} | Average Top1_Accuracy_V:{} | Average Top5_Accuracy_V:{} | Time:{}'.format(
            epoch, train_loss / len(train_loader), top1_accuracy_total_V / len(train_loader.dataset),
                   top5_accuracy_total_V / len(train_loader.dataset), round((time.time() - total_time), 4)))

    with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
        # print('write in-epoch loss at epoch {} | batch {}'.format(epoch,batch_idx))
        file.write('%f\n' % (train_loss / len(train_loader)))

    # save current model
    torch.save(model.state_dict(), result_path + 'model' + str(epoch) + '.pt')


def lr_scheduler(optimizer, init_lr, epoch, lr_decay_iter):
    if epoch % lr_decay_iter:
        return init_lr

    # drop to 0.1*init_lr
    lr = init_lr * 0.1
    optimizer.param_groups[1]['lr'] = lr
    # optimizer.param_groups[2]['lr'] = lr
    # if lr > 0:
    #    optimizer.param_groups[0]['lr'] = lr

    return lr


decay = 4
EPOCHS = decay * 3 + 1

for epoch in range(1, EPOCHS + 1):
    learning_rate = lr_scheduler(optimizer, learning_rate, epoch, decay)
    print(learning_rate)
    train(epoch)





