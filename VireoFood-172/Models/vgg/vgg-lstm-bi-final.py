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
image_path = os.path.join(root_path, image_folder, '/')

file_path = os.path.join(root_path, 'SplitAndIngreLabel/')
ingredient_path = os.path.join(file_path, 'IngreLabel.txt')
glove_path = os.path.join(root_path, 'SplitAndIngreLabel/', 'glove.6B.300d.txt')

train_data_path = os.path.join(file_path, 'TR.txt')
validation_data_path = os.path.join(file_path, 'VAL.txt')
test_data_path = os.path.join(file_path, 'TE.txt')

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

        elif test_data:
            with io.open(test_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            labels = matio.loadmat(file_path + 'test_label.mat')['test_label'][0, :]
            ingredients = matio.loadmat(file_path + 'wordIndicator_test.mat')['wordIndicator_test']

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
    def __init__(self, max_seq=30, num_hidden=4096, num_key_ingre=5, dropout=0.5):
        super(encoder_t, self).__init__()

        # load glove vectors
        wordVector = matio.loadmat(file_path + 'wordVector.mat')['wordVector']
        # add a zero vector on top for padding_idx
        wordVector = np.concatenate([np.zeros((1, wordVector.shape[1])), wordVector], 0)

        self.embedding = nn.Embedding(wordVector.shape[0], wordVector.shape[1], padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(wordVector))

        self.gru = nn.GRU(wordVector.shape[1], num_hidden, dropout=dropout)

        self.num_hidden = num_hidden
        self.num_key_ingre = num_key_ingre
        self.dropout = nn.Dropout(dropout)

        # linear layers for gru attention
        self.ws1 = nn.Linear(num_hidden, num_hidden)
        self.ws2 = nn.Linear(num_hidden, num_key_ingre)
        self.ingre2att = nn.Linear(num_key_ingre, 1)
        self.tanh = nn.Tanh()

        self._initialize_weights()

    def forward(self, y):
        # compute latent vectors
        # indexVector, num_words_per_data, word_label = getIndexVector(y, self.max_seq)
        # indexVector = torch.from_numpy(indexVector).long().cuda(3)
        embed_vector = self.embedding(y)
        embed_vector = embed_vector.permute(1, 0, 2)

        # obtain gru output of hidden vectors
        h0_en = Parameter(torch.zeros((1, y.shape[0], self.num_hidden), requires_grad=True))
        self.gru.flatten_parameters()
        y_embeds, _ = self.gru(embed_vector, h0_en.cuda(3))

        att_y_embeds, multi_attention = self.getAttention(y_embeds)

        return att_y_embeds, multi_attention, y_embeds, embed_vector

    def getAttention(self, y_embeds):
        y_embeds = self.dropout(y_embeds)  # (seq, batch, hidden)
        y_embeds = y_embeds.transpose(0, 1)  # (batch, seq, hidden)

        # compute multi-focus self attention by a two-layer mapping
        # (batch, seq, hidden) -> (batch, seq, hidden) -> (batch, seq, self.num_key_ingre)
        multi_attention = self.ws2(self.tanh(self.ws1(y_embeds)))
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
    def __init__(self, num_hidden=4096, num_glove=300, dropout=0.1):
        super(decoder_t, self).__init__()

        self.hiddenMap = nn.Linear(num_hidden, num_glove)
        self.score = nn.Linear(num_glove + num_hidden, num_glove)
        self.aggregater = nn.Parameter(torch.rand(num_glove, requires_grad=True))

        self.gru = nn.GRU(num_hidden, num_glove, dropout=dropout)

        self.num_glove = num_glove
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

        self._initialize_weights()

    def forward(self, y, context):
        # compute reconstructed vectors
        y = self.dropout(y)  # do a dropout to y, i.e. the output of encoder h_en of size (seq, batch, num_hidden)

        prev_hidden = self.hiddenMap(context).unsqueeze(
            0)  # (batch, num_hidden) -> (batch, num_glove) -> (1, batch, num_glove)
        h_de = []  # store the output hidden vectors of gru
        context_vec = []  # store the context vector, i.e. hidden vectors after attention

        for i in range(0, y.shape[0]):  # for each of the seq's
            current_input = y[i].unsqueeze(0)  # (1, batch, num_hidden)
            current_hidden, prev_hidden = self.gruLoop(current_input,
                                                       prev_hidden)  # NOTE: current_hidden = prev_hidden, we use different notations to clarify their roles

            h_de.append(current_hidden)

            att_weights = self.getAttention(current_hidden, y).unsqueeze(1)  # (batch, 1, seq)
            context = (att_weights.bmm(y.transpose(0, 1)).transpose(0,
                                                                    1))  # (batch, 1, seq) * (batch, seq, num_hidden) = (batch, 1, num_hidden) -> (1, batch, num_hidden)

            context_vec.append(context)

        return torch.cat(h_de, 0), torch.cat(context_vec, 0)

    def gruLoop(self, current_input,
                prev_hidden):  # use this function to avoid a modification of prev_hidden with inplace operation
        output, hidden = self.gru(current_input, prev_hidden)
        return output, hidden

    def getAttention(self, hidden, encoder_outputs):
        tiled_hidden = hidden.repeat(encoder_outputs.data.shape[0], 1,
                                     1)  # both prev_hidden and y have the same size (seq, batch, num_glove)
        energy = F.tanh(self.score(torch.cat([tiled_hidden, encoder_outputs],
                                             2)))  # (Seq, Batch_size, num_glove+num_hidden)->(Seq, Batch_size, num_glove)

        aggregater = self.aggregater.repeat(encoder_outputs.data.shape[1], 1).unsqueeze(1)  # (batch, 1, num_glove)
        energy = torch.bmm(aggregater, energy.permute(1, 2,
                                                      0))  # (batch, 1, num_glove) * (Batch_size, num_glove, Seq) = (batch, 1, seq)

        return self.softmax(energy.squeeze(1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


# entire model
class MyModel(nn.Module):
    def __init__(self, num_key_ingre=5, max_seq=30):
        super(MyModel, self).__init__()
        # network for image channel
        self.encoder = vgg19_bn().cuda(1)

        self.decoder = SegNet().cuda(2)

        # network for ingredient channel
        self.encoder_t = encoder_t().cuda(3)
        self.decoder_t = decoder_t().cuda(3)

        # classifier
        self.classifier_v = nn.Linear(blk_len, 172).cuda(3)
        self.classifier_t = nn.Linear(blk_len, 172).cuda(3)

        # embedding to word indicator vector
        self.one_hot = nn.Linear(4096 + 300, 309).cuda(3)
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self._initialize_weights()

        # domain transfer
        self.trans_img2l = nn.Linear(blk_len, blk_len).cuda(3)
        # self.trans_text2l = nn.Linear(blk_len, blk_len).cuda(3)
        # self.trans_both2l = nn.Linear(blk_len * 2, blk_len).cuda(3)

        # cross-channel recovery
        # self.cross_img2img = nn.Linear(4096, 4096).cuda(3)
        self.cross_img2z = nn.Linear(4096, 4096).cuda(3)
        # self.cross_text2text = nn.Linear(4096, 4096).cuda(3)
        self.cross_text2z = nn.Linear(4096, 4096).cuda(3)
        # self.cross_both2both = nn.Linear(4096 * 2, 4096).cuda(3)
        self.cross_both2z = nn.Linear(4096 * 2, 4096).cuda(3)

        # self.cross_Zimg2Ztext1 = nn.Linear(4096, 4096).cuda(3)
        # self.cross_Zimg2Ztext2 = nn.Linear(4096, 4096).cuda(3)

        self.cross_z2img = nn.Linear(4096, 4096).cuda(3)
        # self.cross_z2img2 = nn.Linear(4096, 4096).cuda(3)

        self.cross_z2text = nn.Linear(4096, 4096).cuda(3)
        # self.cross_z2text2 = nn.Linear(4096, 4096).cuda(3)


        # linear layers to map attended gru vector to re-generate word hidden vectors produced by encoder_t
        self.rec_att2ingre = nn.Linear(1, num_key_ingre).cuda(3)
        self.rec_ingre2seq = nn.Linear(num_key_ingre, max_seq).cuda(3)

    def forward(self, x, y):  # x:image, y:ingredient
        # compute image latent vectors & recons
        # print(x.get_device())
        ##print(x.cuda(1).get_device())
        # print(y.get_device())
        # print(y.cuda(3).get_device())
        x_latent = self.encoder(x.cuda(1))
        x_recon = self.decoder(x_latent.cuda(2))
        x_recon = x_recon.cuda(3)

        # compute ingredient vectors
        att_y_latent, multi_attention, y_embeds, embed_vector = self.encoder_t(y.cuda(3))

        # compute cross-channel regeneration loss
        loss_z, loss_img, loss_text, rec_imgLatent = self.get_cross_channel_loss(x_latent.cuda(3), att_y_latent)

        # compute recovered hidden vectors from encoder
        encoder_outcome_recon = self.getDeAttention(att_y_latent)  # (seq,batch,hidden)

        # compute the predicts of words using recovered hidden vectors
        wordPredict_rec, decoder_y_recon = self.getPredictRec(encoder_outcome_recon, att_y_latent)

        # compute v t predicts in domain adapted space
        predicts, predicts_t, AE = self.get_predicts_with_align(x_latent[:, 0:blk_len].cuda(3),
                                                                att_y_latent[:, 0:blk_len])

        # recon of word embeddings
        y_recon, y_context = self.decoder_t(y_embeds, att_y_latent)  # seq, batch, glove

        att_y = torch.cat((y_context, y_recon), 2)

        # predict words: compute one-hot mappings from word embeddings to word indicators
        wordPredicts = self.one_hot(att_y.permute(1, 0, 2))  # batch, seq, glove -> batch, seq, num_words

        return predicts.cuda(), predicts_t.cuda(), embed_vector.cuda(), multi_attention.cuda(), y_embeds.cuda(), x_recon.cuda(), y_recon.cuda(), wordPredicts.cuda(), x_latent[
                                                                                                                                                                      :,
                                                                                                                                                                      0:blk_len].cuda(), att_y_latent[
                                                                                                                                                                                         :,
                                                                                                                                                                                         0:blk_len].cuda(), encoder_outcome_recon.cuda(), loss_z.cuda(), loss_img.cuda(), loss_text.cuda(), AE.cuda(), wordPredict_rec.cuda(), decoder_y_recon.cuda(), rec_imgLatent.cuda()

    def getPredictRec(self, encoder_outcome_recon, y_latent):
        # Share the same decoder
        decoder_y_recon, y_context = self.decoder_t(encoder_outcome_recon, y_latent)
        att_y = torch.cat((y_context, decoder_y_recon), 2)
        wordPredict_rec = self.one_hot(att_y.permute(1, 0, 2))

        return wordPredict_rec, decoder_y_recon

    def getDeAttention(self, att_y_embeds):
        # att_y_embeds = self.dropout(att_y_embeds) #(batch, hidden)
        att_y_embeds = att_y_embeds.unsqueeze(2)  # (batch, hidden,1)
        # recover attended embeds to its representation of self.num_key_ingre
        att_y_embeds = self.tanh(self.rec_att2ingre(att_y_embeds))  # (batch, hidden, self.num_key_ingre)
        # recover attended embeds to be of length seq
        att_y_embeds = self.rec_ingre2seq(att_y_embeds)  # (batch, hidden, seq)
        return att_y_embeds.permute(2, 0, 1)

    def get_predicts_with_align(self, x_latent, y_latent):
        # compute features in the shared latent domain
        x_latent2l = self.trans_img2l(x_latent)
        # y_latent = self.trans_text2l(y_latent)
        # both_latent = self.trans_both2l(torch.cat((x_latent, y_latent), 1))

        # compute the losses for aligning visual and textual features
        # loss_l2 = torch.sum((x_latent - both_latent) ** 2 + (y_latent - both_latent) ** 2)
        # loss_mean = torch.sum((torch.mean(x_latent,0) - torch.mean(both_latent,0)) ** 2 + (torch.mean(y_latent,0) - torch.mean(both_latent,0)) ** 2)
        # loss_coral = self.get_CORAL_loss(x_latent, both_latent) + self.get_CORAL_loss(y_latent, both_latent)

        kl_loss = nn.KLDivLoss()
        AE = kl_loss(self.log_softmax(x_latent2l),
                     self.softmax(y_latent).detach())  # loss_l2*1e-2 + loss_mean + loss_coral*1e-4
        # print('x_latent2l = {}'.format(self.log_softmax(x_latent2l).data))
        # print('y_latent = {}'.format(self.softmax(y_latent).data))
        # print('AE = {}'.format(AE))

        predicts = self.classifier_v(x_latent2l)
        predicts_t = self.classifier_t(y_latent)

        return predicts, predicts_t, AE

    def get_CORAL_loss(self, x_latent, both_latent):
        # compute covariance matrix for source channel, i.e. x_latent
        x_mean = torch.mean(x_latent, 0, keepdim=True)
        x_cov = x_latent - x_mean
        x_cov = torch.matmul(x_cov.transpose(0, 1), x_cov)

        # compute covariance matrix for target channel, i.e. both_latent
        both_mean = torch.mean(both_latent, 0, keepdim=True)
        both_cov = both_latent - both_mean
        both_cov = torch.matmul(both_cov.transpose(0, 1), both_cov)

        # compute second order loss
        loss = torch.sum((x_cov - both_cov) ** 2)

        return loss

    def get_cross_channel_loss(self, x_latent, y_latent):
        z_img = self.cross_img2z(x_latent)  # self.cross_img2z(self.relu(self.cross_img2img(x_latent)))  # self.tanh()
        z_text = self.cross_text2z(
            y_latent)  # self.cross_text2z(self.relu(self.cross_text2text(y_latent)))  # self.tanh()
        z_both = self.cross_both2z(torch.cat((x_latent, y_latent),
                                             1))  # self.cross_both2z(self.relu(self.cross_both2both(torch.cat((x_latent, y_latent), 1))))  # self.tanh()

        rec_img = self.cross_z2img(z_img)  # self.cross_z2img2(self.relu(self.cross_z2img1(z_img)))  # self.tanh()
        rec_text = self.cross_z2text(z_text)  # self.cross_z2text2(self.relu(self.cross_z2text1(z_text)))  # self.tanh()

        crossLoss_z = torch.sum((z_img - z_both) ** 2 + (z_text - z_both) ** 2 + (z_text - z_img) ** 2)
        crossLoss_img = torch.sum(
            (x_latent - rec_img) ** 2)  # + 9*torch.sum((x_latent[:,0:blk_len] - rec_img[:,0:blk_len]) ** 2)

        crossLoss_text = torch.sum((y_latent - rec_text) ** 2)
        # print('y_latent= {}'.format(y_latent))
        # print('rec_text= {}'.format(rec_text))
        # print('product= {}'.format((y_latent - rec_text) ** 2))
        # print('crossLoss_text= {}'.format(crossLoss_text))


        return crossLoss_z, crossLoss_img, crossLoss_text, rec_img

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
BATCH_SIZE = 16
LOG_INTERVAL = 10
learning_rate = 1e-4
blk_len = 1536

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


# —Model training & testing———————————————————————————————————————————————————————————————————————————————————————————————————————
def get_updateModel():
    model_dict = model.state_dict()

    # update image channel
    pretrained_dict = torch.load(file_path + 'model9-words.pt', map_location='cpu')
    extracted_dict_img = {k[7:]: v for k, v in pretrained_dict.items() if
                          k.startswith('module.encoder.') or k.startswith('module.decoder.')}

    # update lstm channel
    pretrained_dict = torch.load(file_path + 'model-att-bi.pt', map_location='cpu')
    extracted_dict_lstm = {k: v for k, v in pretrained_dict.items() if
                           k.startswith('decoder.')
                           or k.startswith('encoder_t.')
                           or k.startswith('decoder_t.')
                           or k.startswith('trans_')
                           or k.startswith('cross_')
                           }

    model_dict.update(extracted_dict_img)
    model_dict.update(extracted_dict_lstm)

    model.load_state_dict(model_dict)

    #    with io.open(result_path + 'para.txt', 'w', encoding='utf-8') as file:
    #        file.write('img set:{}\n'.format({v for v in extracted_dict_img}))
    #        file.write('lstm set:{}\n'.format({v for v in extracted_dict_lstm}))
    #        file.write('model set:{}\n'.format({v for v in model_dict}))


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
    fc_params = [p for n, p in model.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    non_fc_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
    params = [{'params': fc_params, 'lr': 1e-10}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    optimizer = optim.Adam(params, lr=lr)  # weight_decay=1e-3, lr=lr)

    return optimizer


optimizer = get_optim(learning_rate)

# optim.Adam(model.parameters(), weight_decay=1e-3,
#                      lr=learning_rate)  # .module.parameters(), lr=learning_rate)

# ------------------------------------------------------------------




# Loss
criterion = nn.CrossEntropyLoss()


def getLSTMloss(wordPredicts, ingredients, embed_vector, y_recon, wordPredict_rec):
    lstm_loss = 0
    reclstm_loss = 0
    ingredients = ingredients.numpy()  # word indicators
    label_vector = torch.zeros(309).cuda()  #
    total_num = 0

    for i in range(0, wordPredicts.shape[0]):  # for each batch item
        index = np.where(ingredients[i, :] == 1)[0]  # compute the index of non-zero entries
        for j in range(0, index.shape[0]):  # for each seq item
            total_num += 1
            # label_vector[:] = 0
            # label_vector[index[j]] = 1  # get corresponding label vector
            # lstm_loss += torch.sum((one_hots[i, j, :] - label_vector) ** 2)
            lstm_loss += criterion(wordPredicts[i, j, :].unsqueeze(0),
                                   torch.from_numpy(np.array(index[j])).unsqueeze(0).cuda())
            reclstm_loss += criterion(wordPredict_rec[i, j, :].unsqueeze(0),
                                      torch.from_numpy(np.array(index[j])).unsqueeze(0).cuda())

    decoder_embed_match = torch.sum((embed_vector - y_recon) ** 2)
    return lstm_loss / total_num, decoder_embed_match, reclstm_loss / total_num


def loss_function(predicts_V, predicts_T, labels, data, ingredients, embed_vector, multi_attention, y_embeds, x_recon,
                  y_recon, wordPredicts, x_latent,
                  y_latent, encoder_outcome_recon, loss_z, loss_img, loss_text, AE, wordPredict_rec, decoder_y_recon,
                  rec_imgLatent):
    # image channel loss
    CE_V = criterion(predicts_V, labels - 1) * 20
    CE_T = criterion(predicts_T, labels - 1) * 10

    RE_V = torch.sum((x_recon - data) ** 2) * (1e-2)
    recloss_tag2img = torch.sum((model.decoder(rec_imgLatent.cuda(2)).cuda() - data) ** 2) * (1e-2)

    # ingredient channel loss
    lstm_loss, decoder_embed_match, reclstm_loss = getLSTMloss(wordPredicts, ingredients, embed_vector, y_recon,
                                                               wordPredict_rec)

    lstm_loss *= 1e0
    decoder_embed_match *= 1e0
    reclstm_loss *= 1e0

    encoder_outcome_recon_loss = torch.sum((encoder_outcome_recon - y_embeds) ** 2) * (1e-5)
    decoder_outcome_recon_loss = torch.sum((decoder_y_recon - y_recon) ** 2) * (1e-5)

    # loss for transfer layers
    loss_z *= 1e-3
    loss_img *= 1e-3
    loss_text *= 1e-2

    # l2 norm for feature alignment
    AE = AE * (1e5)  # torch.sum((x_latent - y_latent) ** 2) * (1e-3)

    # constraints on the attention weights of gru encode
    Identity = torch.eye(multi_attention.shape[1]).unsqueeze(0)  # (1,seq,seq)
    Identity = Identity.repeat(multi_attention.shape[0], 1, 1).cuda()  # (batch,seq,seq)
    ones = torch.ones(multi_attention.shape[1], multi_attention.shape[1])
    ones = ones.repeat(multi_attention.shape[0], 1, 1).cuda()  # (batch,seq,seq)

    multi_attention_Transpose = torch.transpose(multi_attention, 1,
                                                2).contiguous().cuda()  # (batch, num_key_ingredient, seq)
    ATT = torch.sum((multi_attention.bmm(multi_attention_Transpose) * (ones - Identity)) ** 2) * (1e3)
    # ATT = torch.sum((multi_attention.bmm(multi_attention_Transpose) - Identity) ** 2) * (1e-2)

    return CE_V, CE_T, RE_V, lstm_loss, decoder_embed_match, AE, ATT, encoder_outcome_recon_loss, decoder_outcome_recon_loss, loss_z, loss_img, loss_text, reclstm_loss, recloss_tag2img


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
        # if batch_idx == 2:
        #    break
        # print('batch %',batch_idx)
        # ---------------------------------------------------------------------------------------------------------------------------------

        start_time = time.time()
        data = Variable(data)
        indexVector = Variable(indexVector)
        if CUDA:
            data = data.cuda()
            labels = labels.cuda()
            indexVector = indexVector.cuda()

        # obtain output from model
        predicts_V, predicts_T, embed_vector, multi_attention, y_embeds, x_recon, y_recon, wordPredicts, x_latent, y_latent, encoder_outcome_recon, loss_z, loss_img, loss_text, AE, wordPredict_rec, decoder_y_recon, rec_imgLatent = model(
            data, indexVector)

        # loss
        CE_V, CE_T, RE_V, lstm_loss, decoder_embed_match, AE, ATT, encoder_outcome_recon_loss, decoder_outcome_recon_loss, loss_z, loss_img, loss_text, reclstm_loss, recloss_tag2img = loss_function(
            predicts_V,
            predicts_T, labels,
            data, ingredients,
            embed_vector,
            multi_attention,
            y_embeds, x_recon,
            y_recon, wordPredicts,
            x_latent, y_latent,
            encoder_outcome_recon,
            loss_z, loss_img, loss_text, AE, wordPredict_rec, decoder_y_recon, rec_imgLatent)

        # optim for myModel with generator
        optimizer.zero_grad()
        loss = CE_V + CE_T + RE_V + lstm_loss + decoder_embed_match + AE + ATT + encoder_outcome_recon_loss + decoder_outcome_recon_loss + loss_z + loss_img + loss_text + reclstm_loss + recloss_tag2img
        loss.backward()
        train_loss += loss.data
        optimizer.step()

        # compute accuracy
        predicts_V = predicts_V.cpu()
        labels = labels.cpu()

        matches_V, hits_V = top_match(predicts_V, labels)
        # top 1 accuracy
        top1_accuracy_total_V += matches_V
        top1_accuracy_cur_V = matches_V / float(labels.size(0))

        # top 5 accuracy
        top5_accuracy_total_V += hits_V
        top5_accuracy_cur_V = hits_V / float(labels.size(0))

        if epoch == 1 and batch_idx == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | RE_V: {:.4f} | recloss_tag2img: {:.4f} | lstm_loss: {:.4f} | reclstm_loss: {:.4f} | decoder_embed_match: {:.4f} | AE: {:.4f} | ATT: {:.4f} | encoder_outcome_recon_loss: {:.4f} |  decoder_outcome_recon_loss: {:.4f} | loss_z: {:.4f} | loss_img: {:.4f} | loss_text: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader),
                    loss.data, CE_V.data, CE_T.data, RE_V.data, recloss_tag2img.data, lstm_loss.data, reclstm_loss.data,
                    decoder_embed_match.data,
                    AE.data, ATT.data, encoder_outcome_recon_loss.data, decoder_outcome_recon_loss.data, loss_z.data,
                    loss_img.data, loss_text.data,
                    top1_accuracy_cur_V, top5_accuracy_cur_V,
                    round((time.time() - start_time), 4),
                    round((time.time() - total_time), 4)))

            with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
                # print('write in-batch loss at epoch {} | batch {}'.format(epoch,batch_idx))
                file.write('%f\n' % (train_loss))

        elif batch_idx % LOG_INTERVAL == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | RE_V: {:.4f} | recloss_tag2img: {:.4f} | lstm_loss: {:.4f} | reclstm_loss: {:.4f} | decoder_embed_match: {:.4f} | AE: {:.4f} | ATT: {:.4f} | encoder_outcome_recon_loss: {:.4f} | decoder_outcome_recon_loss: {:.4f} | loss_z: {:.4f} | loss_img: {:.4f} | loss_text: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader),
                    loss.data, CE_V.data, CE_T.data, RE_V.data, recloss_tag2img.data, lstm_loss.data, reclstm_loss.data,
                    decoder_embed_match.data,
                    AE.data, ATT.data, encoder_outcome_recon_loss.data, decoder_outcome_recon_loss.data, loss_z.data,
                    loss_img.data, loss_text.data,
                    top1_accuracy_cur_V, top5_accuracy_cur_V,
                           round((time.time() - start_time), 4) * LOG_INTERVAL,
                    round((time.time() - total_time), 4)))

        # records current progress for tracking purpose
        with io.open(result_path + 'model_batch_train_loss.txt', 'w', encoding='utf-8') as file:
            file.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | RE_V: {:.4f} | recloss_tag2img: {:.4f} | lstm_loss: {:.4f} | reclstm_loss: {:.4f} | decoder_embed_match: {:.4f} | AE: {:.4f} | ATT: {:.4f} | encoder_outcome_recon_loss: {:.4f} |  decoder_outcome_recon_loss: {:.4f} | loss_z: {:.4f} | loss_img: {:.4f} | loss_text: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader),
                    loss.data, CE_V.data, CE_T.data, RE_V.data, recloss_tag2img.data, lstm_loss.data, reclstm_loss.data,
                    decoder_embed_match.data,
                    AE.data, ATT.data, encoder_outcome_recon_loss.data, decoder_outcome_recon_loss.data, loss_z.data,
                    loss_img.data, loss_text.data,
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
    if lr <= 1e-6:
        optimizer.param_groups[0]['lr'] = lr

    return lr


decay = 4
EPOCHS = decay * 3 + 1

for epoch in range(1, EPOCHS + 1):
    learning_rate = lr_scheduler(optimizer, learning_rate, epoch, decay)
    print(learning_rate)
    train(epoch)




