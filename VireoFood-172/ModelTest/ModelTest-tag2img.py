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

result_path = root_path + 'results_word/'  # 'past_results/' #root_path + 'results2/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

photo_path = root_path + 'test/'
if not os.path.exists(photo_path):
    os.makedirs(photo_path)

# #transform indicator vector of ingredients to that of words
#     #load ingredient indicator of samples
# IngreIndicator = matio.loadmat(photo_path + 'ingredients.mat')['ingredients']
#     #load mapping from ingredient to words (ingredient,words)
# ingre2word_map = matio.loadmat(file_path + 'ingre2word_map.mat')['ingre2word_map']
#     #compute the final word indicator for data
# wordIndicator = IngreIndicator @ ingre2word_map
#     #avoid the case of shared words between ingredients
# wordIndicator[np.where(wordIndicator>1)] = 1
# matio.savemat(photo_path + 'wordIndicator.mat', {'wordIndicator': wordIndicator})

# #obtain input with format suitable for embedding layer of gru
#     #load word indicator vectors
# wordIndicator = matio.loadmat(photo_path + 'wordIndicator.mat')['wordIndicator']
# max_seq = 30 #The maximum number of words
# indexVector = np.zeros((wordIndicator.shape[0],max_seq)) #tore the indexes (in our 309 word table) of words presented in each food item
#     #generate input matrix of size = (batch, seq)
# for i in range(0,wordIndicator.shape[0]): #for each food item
#     print('processing image ' + str(i))
#     #get the index vector
#     index = np.where(wordIndicator[i,:] == 1)[0]
#     #assign the index to our indexVector
#     indexVector[i,0:index.shape[0]] = index[:]
# matio.savemat(photo_path + 'indexVector.mat', {'indexVector': indexVector})


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
        self.classifier = nn.Sequential(
            nn.Linear(512 * ((image_size[0] // (2 ** 5)) ** 2), 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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


# decoder network for image
class SegNet(nn.Module):
    def __init__(self, init_weights=True):
        super(SegNet, self).__init__()

        self.latent_re = nn.Sequential(
            nn.Linear(4096, 512 * ((image_size[0] // (2 ** 5)) ** 2)),
            nn.ReLU(True),
            nn.Dropout(),
        )

        batchNorm_momentum = 0.1

        self.upsample5 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv54d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn54d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.upsample4 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv44d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn44d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv34d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn34d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x5p = self.latent_re(x)
        x5p = x5p.view(x.size(0), 512, image_size[0] // (2 ** 5), image_size[0] // (2 ** 5))

        # Stage 5d
        x5d = self.upsample5(x5p)
        x54d = F.relu(self.bn54d(self.conv54d(x5d)))
        x53d = F.relu(self.bn53d(self.conv53d(x54d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = self.upsample4(x51d)
        x44d = F.relu(self.bn44d(self.conv44d(x4d)))
        x43d = F.relu(self.bn43d(self.conv43d(x44d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = self.upsample3(x41d)
        x34d = F.relu(self.bn34d(self.conv34d(x3d)))
        x33d = F.relu(self.bn33d(self.conv33d(x34d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = self.upsample2(x31d)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = self.upsample1(x21d)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        x_recon = self.sigmoid(x11d)
        return x_recon

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

        return att_y_embeds

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


set_gpu_encoder = 1
set_gpu_decoder = 0
set_gpu_others = 0


# entire model
class MyModel(nn.Module):
    def __init__(self, num_key_ingre=5, max_seq=30):
        super(MyModel, self).__init__()
        # network for image channel
        self.encoder = vgg19_bn().cuda(set_gpu_encoder)
        # self.vgg_map2vec = nn.Linear(512 * ((image_size[0] // (2 ** 5)) ** 2), 4096).cuda(set_gpu_others)
        # self.vgg_linear = nn.Linear(4096, 4096).cuda(set_gpu_others)

        self.decoder = SegNet().cuda(set_gpu_decoder)
        # text channel modules
        # self.encoder_t = encoder_t().cuda(set_gpu_others)

        # classifier
        # self.classifier_v = nn.Linear(blk_len, 172).cuda(set_gpu_others)
        # self.classifier_t = nn.Linear(blk_len, 172).cuda(set_gpu_others)

        # domain transfer
        # self.trans_img2l = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)
        # self.trans_text2l = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)

        # cross channel generation
        # self.cross_img2z = nn.Linear(4096, 4096).cuda(set_gpu_others)
        # self.cross_z2zlinear = nn.Linear(4096, 4096).cuda(set_gpu_others)
        # self.cross_z2text = nn.Linear(4096, 4096).cuda(set_gpu_others)
        # self.cross_text2z = nn.Linear(4096, 4096).cuda(set_gpu_others)
        # self.cross_z2img = nn.Linear(4096, 4096).cuda(set_gpu_others)

        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):  # x:image, y:ingredient
        # compute image latent vectors & recons
        # x_latent_maps = self.encoder(x)
        # x_latent = self.get_latent(x_latent_maps.cuda(set_gpu_others))
        x_latent = self.encoder(x)

        # compute ingredient vectors
        # att_y_latent = self.encoder_t(y)


        # get x_recon
        # x_latent_recon = self.get_x_recon(att_y_latent)

        # get img recon
        img_recon = self.decoder(x_latent.cuda(set_gpu_decoder))

        return img_recon.cuda(set_gpu_others)

    def get_x_recon(self, y_latent):
        x_latent_recon = self.cross_z2img(
            self.dropout(self.relu(self.cross_text2z(
                y_latent))))

        return x_latent_recon

    def get_latent(self, x_latent_maps):
        x_latent = x_latent_maps.view(x_latent_maps.size(0), -1)
        x_latent = self.dropout(self.relu(self.vgg_map2vec(x_latent)))
        x_latent = self.dropout(self.relu(self.vgg_linear(x_latent)))
        return x_latent


# Manual settings———————————————————————————————————————————————————————————————————————————————————————————————————————
# Image Info
no_of_channels = 3
image_size = [256, 256]  # [64,64]

# changed configuration to this instead of argparse for easier interaction
CUDA = 1  # 1 for True; 0 for False
SEED = 1
BATCH_SIZE = 32
LOG_INTERVAL = 10
blk_len = 1536


# —Load input data———————————————————————————————————————————————————————————————————————————————————————————————————————
def readImage(photo_path):
    trans = transforms.ToTensor()
    imgs = torch.zeros(BATCH_SIZE, no_of_channels, image_size[0], image_size[1])
    for i in range(0, BATCH_SIZE):
        # print('Loading image {}...'.format(i))
        imgs[i, :] = trans(Image.open(photo_path + str(i) + '.jpg').convert('RGB'))

    return imgs


# obtain tensor images
images = readImage(photo_path)


# obtain indicator for word presents of samples
# wordIndicator = matio.loadmat(photo_path + 'wordIndicator.mat')['wordIndicator']
# wordIndicator = wordIndicator.astype(np.float32)
# wordIndicator = torch.from_numpy(wordIndicator)
# obtain the indexes of words presented in each sample
# indexVector = matio.loadmat(photo_path + 'indexVector.mat')['indexVector']
# indexVector = indexVector.astype(np.long)
# indexVector = torch.from_numpy(indexVector)
# —Load model———————————————————————————————————————————————————————————————————————————————————————————————————————
def get_updateModel(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k.startswith('encoder.')
                       or k.startswith('decoder.')
                       # or k.startswith('encoder_t.')
                       # or k.startswith('cross_text2z.')
                       # or k.startswith('cross_z2img.')
                       }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


# Model
model = MyModel()
path = result_path + 'model6.pt'
model = get_updateModel(model, path)

# —Model test: image reconstruction———————————————————————————————————————————————————————————————————————————————————————————————————————
model.eval()

# cuda the input
if CUDA:
    images = images.cuda(set_gpu_encoder)
    # wordIndicator = wordIndicator.cuda(set_gpu_others)
    # indexVector = indexVector.cuda(set_gpu_others)

mini_batch_size = 4
num_batch = int(BATCH_SIZE / mini_batch_size)
for i in range(0, num_batch):
    # img_recon = model(images[(i*mini_batch_size):((i+1)*mini_batch_size),:],indexVector[(i*mini_batch_size):((i+1)*mini_batch_size),:])
    img_recon = model(images[(i * mini_batch_size):((i + 1) * mini_batch_size),
                      :])  # model(indexVector[(i*mini_batch_size):((i+1)*mini_batch_size),:])
    print('saving images')
    save_image(torch.cat((images[(i * mini_batch_size):((i + 1) * mini_batch_size), :], img_recon), 0),
               photo_path + 'tag2img' + str(i) + '.jpg', nrow=mini_batch_size)
    # save_image(torch.cat((images[(i*mini_batch_size):((i+1)*mini_batch_size),:],recon_images,tag2img),0), photo_path + 'tag2img'+str(i)+'.jpg', nrow=mini_batch_size)
#
#

# for i in range(0,num_batch):
# get recon_image from original image
# print('Get recon photos...')
# recon_img = getReconImg(model,images[(i*mini_batch_size):((i+1)*mini_batch_size)])
# print('Save recon photos...')
# save_image(recon_img, photo_path + 'recon_img.jpg', nrow=mini_batch_size)
# del recon_img
# print('Get tag-to-photos...')
# a = indexVector[(i*mini_batch_size):((i+1)*mini_batch_size)]
# tag2img = tag2Img(model,indexVector)
# print('Save tag2img...')
# save_image(tag2img, photo_path + 'tag2img.jpg', nrow=mini_batch_size)
# del tag2img



#
#     # produce varied latent vectors
#     batch_imgs = batch_img_producer(latent_imgs)  # 64*64
#
#     # produce reconstructed images
#     recon = model.decoder(np.concatenate([latent_imgs, batch_imgs], 0))
#
#     # move to gpu
#     imgs.to('cuda:3')
#     recon.to('cuda:3')
#
#     # save original and reconstructed imgs
#     # each row shows the 8 images
#     # with right below them the reconstructed output
#     print('Generating images..')
#
#     # generate figure for train images
#     comparison = torch.zeros(64, 3, 256, 256).cuda(3)
#     for i in range(0, 8):
#         if i % 2:
#             comparison[(i * 8):(i * 8) + 8, :] = imgs[(i * 4):(i * 4) + 8, :]
#         else:
#             comparison[(i * 8):(i * 8) + 8, :] = recon[((i // 2) * 8):((i // 2) * 8) + 8, :]
#     save_image(comparison.data, result_path + 'reconstruction_train.jpg', nrow=8)
#     # generate figure for test images
#     for i in range(0, 8):
#         if i % 2:
#             comparison[(i * 8):(i * 8) + 8, :] = imgs[(32 + (i * 4)):(32 + (i * 4) + 8), :]
#         else:
#             comparison[(i * 8):(i * 8) + 8, :] = recon[(32 + (i // 2) * 8):(32 + (i // 2) * 8) + 8, :]
#     save_image(comparison.data, result_path + 'reconstruction_test.jpg', nrow=8)
#
#     # save synthetic images with varied latent values
#     for i in range(0, 32):
#         save_image(recon[(64 + i * 64):(128 + i * 64), :], result_path + 'synthetic_train_' + str(i) + '.jpg',
#                    nrow=8)
#     for i in range(0, 32):
#         save_image(recon[(33 * 64 + i * 64):(34 * 64 + i * 64), :],
#                    result_path + 'synthetic_test_' + str(i) + '.jpg', nrow=8)
#
# # —Model testing———————————————————————————————————————————————————————————————————————————————————————————————————————
#
#
# def batch_img_producer(latent_imgs):
#     # set parameters
#     rng = np.arange(-5, 5 + (10 / 8), (10 / 8))  # set variable varies in [-3,3] with step 1
#     num_feature_groups = len(rng)  # how many groups of features to change
#     feature_group_size = latent_imgs.shape[1] // num_feature_groups  # number of features per feature group
#     rng = torch.tensor(rng).view(-1, 1)  # make rng a column vector
#     batch_rng = rng.repeat(1, feature_group_size)  # repeat to change the feature values of multiple latent vectors
#
#     # create tensor to store varied latent vectors
#     batch_imgs = torch.zeros(latent_imgs.shape[0] * len(rng) * num_feature_groups, latent_imgs.shape[1])
#
#     # processing each latent vector one by one
#     for j in range(0, latent_imgs.shape[0]):
#         x_latent = latent_imgs[j, :]
#         # duplicate latent vector with individually changed feature group values
#         batch_img = x_latent.repeat(len(rng) * num_feature_groups, 1)
#
#         for i in range(0, num_feature_groups):
#             if i == num_feature_groups - 1:
#                 batch_img[(i * len(rng)):((i + 1) * len(rng)), i * feature_group_size:] = rng.repeat(1, x_latent.shape[
#                     1] - i * feature_group_size)[:]
#                 break
#
#             batch_img[i * len(rng):(i + 1) * len(rng), i * feature_group_size:(i + 1) * feature_group_size] = batch_rng[
#                                                                                                               :]  # vary the values of i-th feature group for i-th image batch (in total num_features batches)
#         # store the varied values
#         batch_imgs[j * len(rng) * num_feature_groups:(j + 1) * len(rng) * num_feature_groups:] = batch_img[:]
#
#     return batch_imgs
#
#
#
#
