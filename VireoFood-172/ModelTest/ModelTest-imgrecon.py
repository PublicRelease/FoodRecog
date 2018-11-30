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

past_result_path = root_path + 'past_results/Atemp-recon1/'

result_path = past_result_path  # root_path + 'results/'  # 'past_results/' #root_path + 'results2/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

photo_path = root_path + 'test/'

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
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def Deconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    if (stride - 2) == 0:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, output_padding=1, bias=False)
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)


class DeBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DeBasicBlock, self).__init__()
        self.conv1 = Deconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = Deconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def Deconv_Bottleneck(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    if (stride - 2) == 0:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, output_padding=1, bias=False)
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)


class DeBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DeBottleneck, self).__init__()
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Deconv_Bottleneck(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.ConvTranspose2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, deblock, layers, num_classes=172):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # define resnet encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0])  # 64-64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 64-128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 128-256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 256-512
        self.avgpooling = nn.AvgPool2d(image_size[0] // (2 ** 5), stride=1)
        # get latent representation
        self.latent = nn.Linear(512 * block.expansion, 500)

        # classifier
        self.classifier1 = nn.Linear(blk_len, num_classes)

        # define resnet decoder
        self.latent_re = nn.Linear(latent_len, 512 * block.expansion)
        self.layer5 = self._make_Delayer(deblock, 256, layers[3], stride=2)  # 512-256
        self.layer6 = self._make_Delayer(deblock, 128, layers[3], stride=2)  # 256-128
        self.layer7 = self._make_Delayer(deblock, 64, layers[3], stride=2)  # 128-64
        self.layer8 = self._make_Delayer(deblock, 64, layers[3], stride=1)  # 64-64
        self.unmaxpool = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)
        self.deconv9 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1,
                                          bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_Delayer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, output_padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encoder(self, x):  # convolve images
        x = self.conv1(x)  # ／2
        x = self.bn1(x)
        x = self.relu(x)
        [x, a] = self.maxpool(x)  # ／2
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)  # ／2
        # print(x.shape)
        x = self.layer3(x)  # ／2
        # print(x.shape)
        x = self.layer4(x)  # ／2
        # print(x.shape)
        x = self.avgpooling(x)  # (1x1)
        # print(x.shape)

        x = x.view(x.size(0), -1)
        x_latent = self.latent(x)

        return x_latent, a

    def decoder(self, x_latent, a):
        x = self.latent_re(x_latent)
        x = x.view(x.shape[0], 512, 1, 1)
        # print(x.shape)
        x = F.upsample(x, scale_factor=image_size[0] // (2 ** 5), mode='nearest')
        # print(x.shape)
        x = self.layer5(x)
        # print(x.shape)
        x = self.layer6(x)
        # print(x.shape)
        x = self.layer7(x)
        # print(x.shape)
        x = self.layer8(x)
        # print(x.shape)
        x = self.unmaxpool(x, a)
        # print(x.shape)
        x = self.deconv9(x)
        # print(x.shape)
        x = self.sigmoid(x)
        return x

    def forward(self, x):  # x:image y:ingredient
        x_latent, a = self.encoder(x)

        # predicts = self.classifier1(x_latent[:,0:blk_len])

        x = self.decoder(x_latent, a)

        return x, x_latent, a


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, DeBasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, DeBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


# Manual settings———————————————————————————————————————————————————————————————————————————————————————————————————————
# Image Info
no_of_channels = 3
image_size = [256, 256]  # [64,64]

# changed configuration to this instead of argparse for easier interaction
CUDA = 1  # 1 for True; 0 for False
SEED = 1
BATCH_SIZE = 32
LOG_INTERVAL = 10
latent_len = 500
blk_len = 500  # int(latent_len * 3 / 8)


# —Load input data———————————————————————————————————————————————————————————————————————————————————————————————————————
def readImage(photo_path):
    trans = transforms.ToTensor()
    imgs = torch.zeros(BATCH_SIZE, no_of_channels, image_size[0], image_size[1])
    for i in range(0, BATCH_SIZE):
        # print('Loading image {}...'.format(i))
        imgs[i, :] = trans(Image.open(photo_path + str(i) + '.jpg').convert('RGB'))

    return imgs.view(BATCH_SIZE, 3, 256, 256)


# obtain tensor images
# images = readImage(photo_path)


# obtain indicator for word presents of samples
# wordIndicator = matio.loadmat(photo_path + 'wordIndicator.mat')['wordIndicator']
# wordIndicator = wordIndicator.astype(np.float32)
# wordIndicator = torch.from_numpy(wordIndicator)
# obtain the indexes of words presented in each sample
# indexVector = matio.loadmat(photo_path + 'indexVector.mat')['indexVector']
# indexVector = indexVector.astype(np.long)
# indexVector = torch.from_numpy(indexVector)
# —Load model———————————————————————————————————————————————————————————————————————————————————————————————————————
def get_updateModel(path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if True
                       # k.startswith('encoder.')
                       # or k.startswith('decoder.')
                       # or k.startswith('latent_x.')
                       # or k.startswith('encoder_t.')
                       # or k.startswith('cross_text2z.')
                       # or k.startswith('cross_z2img.')
                       }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


# Model
model = resnet18()
model = nn.DataParallel(model)
model = model.cuda()
model.eval()
# path = result_path + 'model19.pt' #result_path + 'model11.pt'
# model = get_updateModel(model, path)

# —Model test: image reconstruction———————————————————————————————————————————————————————————————————————————————————————————————————————




# cuda the input
# if CUDA:
#    images = images.cuda(set_gpu_encoder)
# wordIndicator = wordIndicator.cuda(set_gpu_others)
# indexVector = indexVector.cuda(set_gpu_others)


# mini_batch_size = 32
# num_batch = int(BATCH_SIZE / mini_batch_size)
# begin = 1
# end = 13
# for i in range(begin, end+1):
#     for j in range(0, num_batch):
#         path = result_path + 'model'+ str(i)+'.pt'
#         model = get_updateModel(model, path)
#         # img_recon = model(images[(i*mini_batch_size):((i+1)*mini_batch_size),:],indexVector[(i*mini_batch_size):((i+1)*mini_batch_size),:])
#         img_recon = model(images[(j*mini_batch_size):((j+1)*mini_batch_size),:])
#                           #[(i * mini_batch_size):((i + 1) * mini_batch_size),
#                           #:])  # model(indexVector[(i*mini_batch_size):((i+1)*mini_batch_size),:])
#         print('saving images')
#         #save_image(torch.cat((images, img_recon), 0),
#         #           result_path + 'tag2img' + str(i+1) + '.jpg', nrow=8)
#         save_image(torch.cat((images[(j*mini_batch_size):((j+1)*mini_batch_size),:],img_recon),0), result_path + str(j+1)+'_'+'model'+str(i)+'.jpg', nrow=8)
#
#

set_gpu_encoder = 1
set_gpu_decoder = 2
set_gpu_img = 3


def tester(epoch):
    # create reconstruction tensor
    comparison = torch.zeros(64, 3, 256, 256)
    if CUDA:
        comparison = comparison.cuda(set_gpu_img)
    trans = transforms.ToTensor()
    # process test images one by one
    for i in range(0, 32):
        print('process img {}'.format(i))
        # read the image tensor
        test_img = trans(Image.open(photo_path + str(i) + '.jpg').convert('RGB')).view(1, 3, 256, 256)
        # obtain latent space
        # x_latent, a = model.encoder(test_img.cuda(set_gpu_encoder))#.module.encoder_V(test_img)
        # produce reconstructed img
        test_img = test_img.cuda()
        recon, x_latent, a = model(test_img)  # .module.decoder_V(x_latent, a)
        # save the original and reconstructed imgs to tensor
        # each row shows the 8 images
        # with right below them the reconstructed output

        x_latent = x_latent.cuda(set_gpu_img)
        recon = recon.cuda(set_gpu_img)
        a = a.cuda(set_gpu_img)

        row = (i + 1) // 8  # the row index for this img
        mod = (i + 1) % 8
        if not mod:
            row -= 1
            col = 8
        else:
            col = (i + 1) % 8

        row *= 2

        comparison[row * 8 + col - 1, :] = test_img
        comparison[(row + 1) * 8 + col - 1, :] = recon
        del test_img
        del recon

        # produce synthetic images with varied latent values
        # modify latent vector
        start = -5
        end = 5
        step = (end - start) / 8
        rng = np.arange(start, end, step)  # set variable varies in [-3,3] with step 1
        num_features = len(rng)  # change the first seven features
        batch_image = batch_img_producer(x_latent, rng, num_features)

        # para_decoder = model.module.decoder.cuda(set_gpu_img) #nn.DataParallel(model.decoder.cuda())
        # produce reconstructed images
        recon_img = model.module.decoder(batch_image.cuda(), a.repeat(len(rng) * num_features, 1, 1,
                                                                      1).cuda())  # .module.decoder_V(batch_image, a.repeat(len(rng) * num_features, 1, 1, 1))
        del batch_image
        save_image(recon_img, result_path + 'synthetic_test_' + str(i) + '.jpg', nrow=len(rng))
        del recon_img

        # save results to the folder, note that should save to the shared folder in docker image, and view files in local folder
    print('Generating images..')
    save_image(comparison.data, result_path + str(epoch) + '_reconstruction_test.jpg', nrow=8)


def batch_img_producer(x_latent, rng, num_feature_groups):
    # duplicate latent vector with individually changed feature value
    batch_img = x_latent.repeat(len(rng) * num_feature_groups, 1)
    feature_group_size = x_latent.shape[1] // num_feature_groups
    rng = torch.tensor(rng).view(-1, 1)
    batch_rng = rng.repeat(1, feature_group_size)

    for i in range(0, num_feature_groups):
        if i == num_feature_groups - 1:
            batch_img[(i * len(rng)):((i + 1) * len(rng)), i * feature_group_size:] = rng.repeat(1, x_latent.shape[
                1] - i * feature_group_size)[:]
            break

        batch_img[i * len(rng):(i + 1) * len(rng), i * feature_group_size:(i + 1) * feature_group_size] = batch_rng[
                                                                                                          :]  # vary the values of i-th feature group for i-th image batch (in total num_features batches)

    return batch_img


start = 12
end = 12

for i in range(start, end + 1):
    path = result_path + 'model' + str(i) + '.pt'  # result_path + 'model11.pt'
    get_updateModel(path)
    tester(i)



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
