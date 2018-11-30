import io
import scipy.io as matio
import os
import os.path
import numpy as np
from PIL import Image
import time
import random

import torch
import torch.utils.data
import torch.nn.parallel as para
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo

# —Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————
root_path = '/mnt/FoodRecog/'  # /home/lily/Desktop/food/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
image_folder = 'ready_chinese_food'  # scaled_images ready_chinese_food
image_path = os.path.join(root_path, image_folder, '/')

file_path = os.path.join(root_path, 'SplitAndIngreLabel/')
ingredient_path = os.path.join(file_path, 'IngreLabel.txt')

train_data_path = os.path.join(file_path, 'TR.txt')
validation_data_path = os.path.join(file_path, 'VAL.txt')
test_data_path = os.path.join(file_path, 'TE.txt')

premodel_path1 = root_path + 'past_results/final-resnet-ingre-step3/'
premodel_path2 = root_path + 'past_results/final-resnet-ingre-step3-ingre/'

result_path = root_path + 'results_resnet-ingre-step4/'
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
            ingredients = matio.loadmat(file_path + 'ingredient_train_feature.mat')['ingredient_train_feature'][0:66071,
                          :]

            with io.open(validation_data_path, encoding='utf-8') as file:
                path_to_images1 = file.read().split('\n')
            labels1 = matio.loadmat(file_path + 'validation_label.mat')['validation_label']
            ingredients1 = matio.loadmat(file_path + 'ingredient_val_feature.mat')[
                               'ingredient_val_feature'][0:11016, :]

            path_to_images = path_to_images + path_to_images1
            labels = np.concatenate([labels, labels1], 1)[0, :]
            ingredients = np.concatenate([ingredients, ingredients1], 0)
        elif test_data:
            with io.open(test_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            labels = matio.loadmat(file_path + 'test_label.mat')['test_label'][0, :]
            ingredients = matio.loadmat(file_path + 'ingredient_test_feature.mat')['ingredient_test_feature'][0:33154,
                          :]
        else:
            with io.open(validation_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            labels = matio.loadmat(file_path + 'validation_label.mat')['validation_label'][0, :]
            ingredients = matio.loadmat(file_path + 'ingredient_validation_feature.mat')[
                'ingredient_validation_feature']

        ingredients = ingredients.astype(np.float32)
        self.path_to_images = path_to_images
        self.labels = labels
        self.ingredients = ingredients
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
        return img, label, ingredient

    def __len__(self):
        return len(self.path_to_images)


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
    def __init__(self, block, layers, num_classes=172):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # define resnet encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # , return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0])  # 64-64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 64-128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 128-256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 256-512
        self.avgpooling = nn.AvgPool2d(image_size[0] // (2 ** 5), stride=1)

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

    def forward(self, x):  # x:image y:ingredient

        x = self.conv1(x)  # ／2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # ／2    [x, a]
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

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


class deResNet(nn.Module):
    def __init__(self, block, layers, num_classes=172):
        self.inplanes = latent_len
        super(deResNet, self).__init__()

        # define resnet decoder
        self.latent_re = nn.Linear(latent_len, 512 * block.expansion)
        self.layer5 = self._make_Delayer(block, 256, layers[3], stride=2)  # 512-256
        self.layer6 = self._make_Delayer(block, 128, layers[3], stride=2)  # 256-128
        self.layer7 = self._make_Delayer(block, 64, layers[3], stride=2)  # 128-64
        self.layer8 = self._make_Delayer(block, 64, layers[3], stride=1)  # 64-64
        # self.deconv9 = nn.ConvTranspose2d(64* block.expansion, 3, kernel_size=7, stride=2, padding=3, output_padding=1,
        #                                  bias=False)
        self.deconv9 = nn.ConvTranspose2d(64 * block.expansion, 64, kernel_size=1, bias=False)

        self.unmaxpool = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, output_padding=1, padding=1)

        # self.unmaxpool = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)
        self.deconv10 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1,
                                           bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

    def forward(self, x_latent):
        x = self.latent_re(x_latent)
        x = x.view(x.shape[0], latent_len, 1, 1)
        # print(x.shape)
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
        x = self.deconv9(x)
        x = self.unmaxpool(x)
        # print(x.shape)
        x = self.deconv10(x)
        # print(x.shape)
        x = self.sigmoid(x)
        return x


def deresnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = deResNet(DeBasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def deresnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = deResNet(DeBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


# encoder network for textual channel
class encoder_t(nn.Module):
    def __init__(self, num_ingre_feature=353):
        super(encoder_t, self).__init__()

        # define ingredient encoder for 353 input features

        self.nn1 = nn.Linear(num_ingre_feature, num_ingre_feature)
        self.nn2 = nn.Linear(num_ingre_feature, latent_len)

        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout()

    def forward(self, y):
        # compute latent vectors
        y = self.relu(self.nn1(y))
        y = self.nn2(y)

        return y


# class decoder_t(nn.Module):
#     def __init__(self, num_ingre_feature=353):
#         super(decoder_t, self).__init__()
#
#         # define ingredient decoder
#         self.nn3 = nn.Linear(latent_len, num_ingre_feature)
#         self.nn4 = nn.Linear(num_ingre_feature, num_ingre_feature)
#
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, y):
#         # compute latent vectors
#         y = self.relu(self.nn3(y))
#         y = self.sigmoid(self.nn4(y))
#
#         return y


set_gpu_encoder = 3
set_gpu_decoder = 2
set_gpu_others = 1
set_gpu_d = 0


# entire model
class MyModel(nn.Module):
    def __init__(self, num_classes=172):
        super(MyModel, self).__init__()
        # network for image channel
        # self.encoder = resnet50(pretrained=True).cuda(set_gpu_encoder)
        self.decoder = deresnet50().cuda(set_gpu_decoder)
        self.img_feature = resnet50(pretrained=True).cuda(set_gpu_encoder)

        # network for ingredient channel
        # encoder
        self.encoder_t = encoder_t().cuda(set_gpu_others)
        # self.decoder_t = decoder_t().cuda(set_gpu_others)

        # get latent(disentangled) representation layers
        # self.latent_x = nn.Linear(latent_len, latent_len).cuda(set_gpu_others)
        self.latent_y = nn.Linear(latent_len, latent_len).cuda(set_gpu_others)

        # cross mapping
        self.cross_t2img1 = nn.Linear(latent_len, latent_len).cuda(set_gpu_others)
        self.cross_t2img2 = nn.Linear(latent_len, latent_len).cuda(set_gpu_others)

        # get partial heterogeneous transfer
        # self.trans_x1 = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)
        # self.trans_x2 = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)
        # self.trans_y1 = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)
        # self.trans_y2 = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)

        # shared classifier
        # self.classifier = nn.Linear(blk_len, num_classes).cuda(set_gpu_others)


        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()
        # self.log_softmax = nn.LogSoftmax()

    def forward(self, x, y):  # x:image y:ingredient
        # get x_latent ready for disentanglement
        # x_latent = self.get_x_latent(x)
        # get y_latent ready for disentanglement
        y_latent = self.get_y_latent(y)

        # get partial heterogeneous transfer space learning
        # predicts_v, predicts_t, AE_l2, AE_kl = self.get_predicts_with_align(x_latent, y_latent)

        # get x recon
        # x_recon = self.decoder(x_latent.cuda(set_gpu_decoder), a.cuda(set_gpu_decoder))

        # get feature level img recon loss
        # recon_feature_loss = self.get_img_feature(x, x_recon)

        # get y recon
        # y_recon = self.decoder_t(y_latent)

        # get x_latent_recon
        x_latent_recon = self.cross_t2img2(self.relu(self.cross_t2img1(y_latent)))

        # get recon img using y_latent
        img_recon = self.get_img_recon(x_latent_recon)

        # feature loss for img recon
        recon_feature_loss = self.get_img_feature(x, img_recon)

        return img_recon.cuda(set_gpu_others), recon_feature_loss.cuda(
            set_gpu_others)  # , x_latent.cuda(set_gpu_others), x_latent_recon.cuda(
        # set_gpu_others)

    def get_img_recon(self, x_latent):
        # print(x_latent.shape)
        # print(a.shape)
        x_recon = self.decoder(x_latent.cuda(set_gpu_decoder))
        return x_recon

    def get_img_feature(self, x, x_recon):
        x_fea = self.img_feature(x.cuda(set_gpu_encoder))
        x_recon_fea = self.img_feature(x_recon.cuda(set_gpu_encoder))

        recon_feature_loss = torch.sum((x_fea - x_recon_fea) ** 2) / (x_fea.shape[0] * x_fea.shape[1])

        return recon_feature_loss

    # def get_predicts_with_align(self, x_latent, y_latent):
    #     # compute features in the transferred latent domain
    #     # image channel
    #     x_trans = self.trans_x2(self.relu(self.trans_x1(
    #         x_latent[:, 0:blk_len])))
    #
    #     # text channel
    #     y_trans = self.trans_y2(self.relu(self.trans_y1(
    #         y_latent[:, 0:blk_len])))
    #
    #     # partial heterogeneous mapping
    #     AE_l2 = torch.sum((x_trans - y_trans) ** 2) / (x_trans.shape[0] * x_trans.shape[1])
    #     AE_kl = kl_loss(self.log_softmax(x_trans),
    #                     self.softmax(y_trans).detach())
    #
    #     # get predicts
    #     predicts_v = self.classifier(x_trans)
    #     predicts_t = self.classifier(y_trans)
    #
    #     return predicts_v, predicts_t, AE_l2, AE_kl

    def get_x_latent(self, x):
        x_vector = self.encoder(x.cuda(set_gpu_encoder))
        x_vector = x_vector.view(x.size(0), -1).cuda(set_gpu_others)
        x_latent = self.latent_x(x_vector)
        return x_latent

    def get_y_latent(self, y):
        y_latent = self.encoder_t(y.cuda(set_gpu_others))
        y_latent = self.latent_y(y_latent)
        return y_latent


class Discriminator(nn.Module):
    def __init__(self, latent=512):
        super(Discriminator, self).__init__()
        self.latent = resnet18(pretrained=True)
        self.linear = nn.Linear(latent, latent)
        self.classifier = nn.Linear(latent, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        latent = self.latent(x)
        linear = self.relu(self.linear(latent.view(latent.size(0), -1)))
        judge = self.sigmoid(self.classifier(linear))
        return judge


# —Manual settings———————————————————————————————————————————————————————————————————————————————————————————————————————
# Image Info
no_of_channels = 3
image_size = [256, 256]  # [64,64]

# changed configuration to this instead of argparse for easier interaction
CUDA = 1  # True
SEED = 1
BATCH_SIZE = 32
LOG_INTERVAL = 10
learning_rate = 1e-3
latent_len = 2048
blk_len = int(latent_len * 3 / 8)

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}

# Download or load downloaded MNIST dataset
# shuffle data at every epoch

train_loader = torch.utils.data.DataLoader(
    FoodData(train_data=True, test_data=False,
             transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    FoodData(train_data=False, test_data=True,
             transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)


# —Model training & testing———————————————————————————————————————————————————————————————————————————————————————————————————————
def get_updateModel():
    model_dict = model.state_dict()

    # update image channel
    pretrained_dict = torch.load(premodel_path1 + 'model19.pt', map_location='cpu')
    extracted_dict_img = {k: v for k, v in pretrained_dict.items() if
                          # k.startswith('encoder.')
                          k.startswith('decoder.')
                          # or k.startswith('latent_x')
                          }

    pretrained_dict = torch.load(premodel_path2 + 'model19.pt', map_location='cpu')
    extracted_dict_text = {k: v for k, v in pretrained_dict.items() if
                           k.startswith('encoder_t.')
                           or k.startswith('latent_y')
                           }

    model_dict.update(extracted_dict_img)
    model_dict.update(extracted_dict_text)
    model.load_state_dict(model_dict)
    return model


lr_freeze = 0
lr_finetune = 1e-6
lr = 1e-3


def get_optim():
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    freeze_params = [p for k, p in model.named_parameters() if
                     k.startswith('img_feature.')
                     # or k.startswith('decoder.')
                     # or k.startswith('encoder_t.')
                     # or k.startswith('decoder_t')
                     # or k.startswith('latent_')
                     ]
    finetune_params = [p for k, p in model.named_parameters() if
                       k.startswith('decoder.')
                       or k.startswith('encoder_t.')
                       or k.startswith('latent_y.')
                       # or k.startswith('decoder.')
                       # or k.startswith('encoder_t')
                       # or k.startswith('decoder_t')
                       # or k.startswith('classifier')
                       ]
    non_frozen_params = [p for k, p in model.named_parameters() if
                         k.startswith('cross_')
                         # or k.startswith('decoder.')
                         # or k.startswith('trans_y')
                         # and not n.startswith('encoder_t')
                         # and not n.startswith('decoder_t')
                         # and not n.startswith('img_feature')
                         ]

    params = [{'params': freeze_params, 'lr': lr_freeze}, {'params': finetune_params, 'lr': lr_finetune},
              {'params': non_frozen_params}]
    # params = [{'params': non_frozen_params}]
    # for param in frozen_params:
    #    param.requires_grad = False

    optimizer = optim.Adam(params, lr=lr)  # weight_decay=1e-3, lr=lr)
    return optimizer


model = MyModel()
model = get_updateModel()
optimizer = get_optim()

D = Discriminator().cuda(set_gpu_d)
D_optimizer = optim.Adam(D.parameters(), lr=lr * 0.1)  # weight_decay =1e-3,

criterion = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss()


def loss_function(img_recon, data):
    RE_V = torch.sum((data.cuda(set_gpu_others) - img_recon) ** 2) / (
    data.shape[0] * image_size[0] * image_size[1]) * 1e2

    # cross_recon_loss = torch.sum((x_latent - x_latent_recon) ** 2) / (data.shape[0] * x_latent.shape[1]) * 1e0

    # generate distortion for label values
    rand_gfake = torch.ones(data.shape[0]) + (6 * torch.rand(data.shape[0]) - 3) * 0.1
    rand_dreal = torch.ones(data.shape[0]) + (6 * torch.rand(data.shape[0]) - 3) * 0.1
    rand_dfake = 0.3 * torch.rand(data.shape[0])

    G_fake = torch.sum((D(img_recon.cuda(set_gpu_d)) - rand_gfake.cuda(set_gpu_d)) ** 2) / data.shape[0] * 1e0
    D_real = torch.sum((D(data.cuda(set_gpu_d)) - rand_dreal.cuda(set_gpu_d)) ** 2) / data.shape[0] * 1e0
    D_fake = (torch.sum(D(img_recon.detach().cuda(set_gpu_d)) - rand_dfake.cuda(set_gpu_d)) ** 2) / data.shape[0] * 1e-3

    # G_fake = criterion(D(img_recon.cuda(set_gpu_d)), torch.ones(data.shape[0],dtype = torch.long).cuda(set_gpu_d)) * 1e1
    # D_real = criterion(D(data.cuda(set_gpu_d)), torch.ones(data.shape[0],dtype = torch.long).cuda(set_gpu_d)) * 1e1
    # D_fake = criterion(D(img_recon.detach().cuda(set_gpu_d)), torch.zeros(data.shape[0], dtype = torch.long).cuda(set_gpu_d)) * 1e1



    return RE_V, G_fake.cuda(set_gpu_others), D_real.cuda(set_gpu_others), D_fake.cuda(set_gpu_others)


def top_match(predicts, labels):
    sorted_predicts = predicts.cpu().data.numpy().argsort()
    top1_labels = sorted_predicts[:, -1:][:, 0]
    match = float(sum(top1_labels == (labels - 1)))

    top5_labels = sorted_predicts[:, -5:]
    hit = 0
    for i in range(0, labels.size(0)):
        hit += (labels[i] - 1) in top5_labels[i, :]

    return match, hit


def top_n_acc(predicts_v, predicts_t, labels):
    matches_V, hits_V = top_match(predicts_v, labels)
    matches_T, hits_T = top_match(predicts_t, labels)
    # top 1 accuracy
    # top1_accuracy_total_V += matches_V
    top1_accuracy_cur_V = matches_V / float(labels.size(0))
    # top1_accuracy_total_T += matches_T
    top1_accuracy_cur_T = matches_T / float(labels.size(0))
    # top 5 accuracy
    # top5_accuracy_total_V += hits_V
    top5_accuracy_cur_V = hits_V / float(labels.size(0))
    # top5_accuracy_total_T += hits_T
    top5_accuracy_cur_T = hits_T / float(labels.size(0))

    return top1_accuracy_cur_V, top1_accuracy_cur_T, top5_accuracy_cur_V, top5_accuracy_cur_T


def train(epoch):
    # toggle model to train mode
    print('Training starts..')

    model.train()
    D.train()

    # change eval mode for frozen nets
    model.img_feature.eval()

    train_loss = 0
    # top1_accuracy_total_V = 0
    # top5_accuracy_total_V = 0
    # top1_accuracy_total_T = 0
    # top5_accuracy_total_T = 0
    total_time = time.time()

    for batch_idx, (data, labels, ingredients) in enumerate(
            train_loader):  # ---------------------------------------------------------------------------------------------------------------------------------
        # for effective code debugging
        # if batch_idx == 1:
        # break
        # print('batch %',batch_idx)         #---------------------------------------------------------------------------------------------------------------------------------
        # for effective code debugging
        start_time = time.time()
        data = Variable(data)
        if CUDA:
            data = data.cuda(set_gpu_encoder)
        ingredients = Variable(ingredients)

        optimizer.zero_grad()
        D_optimizer.zero_grad()

        # obtain output from model
        img_recon, recon_feature_loss = model(data, ingredients)

        # calculate scalar loss
        RE_V, G_fake, D_real, D_fake = loss_function(img_recon, data)
        recon_feature_loss *= 1e2
        loss = RE_V + recon_feature_loss + G_fake + D_real + D_fake
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        D_optimizer.step()

        if epoch == 1 and batch_idx == 0:

            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | RE_V: {:.4f} | recon_feature_loss: {:.4f} | G_fake: {:.4f} | D_real: {:.4f} | D_fake: {:.4f} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.data,
                    RE_V, recon_feature_loss, G_fake, D_real, D_fake,
                    round((time.time() - start_time), 4),
                    round((time.time() - total_time), 4)))

            with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
                # print('write in-batch loss at epoch {} | batch {}'.format(epoch,batch_idx))
                file.write('%f\n' % (train_loss))

        elif batch_idx % LOG_INTERVAL == 0:

            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | RE_V: {:.4f} | recon_feature_loss: {:.4f} | G_fake: {:.4f} | D_real: {:.4f} | D_fake: {:.4f} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.data,
                    RE_V, recon_feature_loss, G_fake, D_real, D_fake,
                    round((time.time() - start_time) * LOG_INTERVAL, 4),
                    round((time.time() - total_time), 4)))

            # records current progress for tracking purpose
            with io.open(result_path + 'model_batch_train_loss.txt', 'w', encoding='utf-8') as file:
                file.write(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | RE_V: {:.4f} | recon_feature_loss: {:.4f} | G_fake: {:.4f} | D_real: {:.4f} | D_fake: {:.4f} | Time:{} | Total_Time:{}'.format(
                        epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                               100. * (batch_idx + 1) / len(train_loader), loss.data,
                        RE_V, recon_feature_loss, G_fake, D_real, D_fake,
                        round((time.time() - start_time) * LOG_INTERVAL, 4),
                        round((time.time() - total_time), 4)))
                # ---------------------------------------------------------------------------------------------------------------------------------
                # for effective code debugging
                # break
                # ---------------------------------------------------------------------------------------------------------------------------------
                #    print(
                #        '====> Epoch: {} | Average loss: {:.4f} | Average Top1_Accuracy_V:{} | Average Top5_Accuracy_V:{} | Average Top1_Accuracy_T:{} | Average Top5_Accuracy_T:{} | Time:{}'.format(
                #            epoch, train_loss / len(train_loader), top1_accuracy_total_V / len(train_loader.dataset),
                #                   top5_accuracy_total_V / len(train_loader.dataset), top1_accuracy_total_T / len(train_loader.dataset),
                #                   top5_accuracy_total_T / len(train_loader.dataset), round((time.time() - total_time), 4)))

    with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
        # print('write in-epoch loss at epoch {} | batch {}'.format(epoch,batch_idx))
        file.write('%f\n' % (train_loss / len(train_loader)))

    # save current model
    torch.save(model.state_dict(), result_path + 'model' + str(epoch) + '.pt')


def lr_scheduler(lr_finetune, lr, epoch, lr_decay_iter):
    if epoch == 2:  # allow finetuning when non-frozen parameters are stable
        lr_finetune = 1e-5
        optimizer.param_groups[1]['lr'] = lr_finetune

    if epoch % lr_decay_iter:
        return lr_finetune, lr

    # drop to 0.1*init_l
    lr *= 0.1
    optimizer.param_groups[2]['lr'] = lr
    D_optimizer.param_groups[0]['lr'] = lr

    if lr < 2e-5:
        lr_finetune *= 0.1
        optimizer.param_groups[1]['lr'] = lr_finetune

    return lr_finetune, lr


decay = 6
EPOCHS = decay * 3 + 1

for epoch in range(1, EPOCHS + 1):
    lr_finetune, lr = lr_scheduler(lr_finetune, lr, epoch, decay)
    print('lr_finetune: {}'.format(lr_finetune))
    print('lr: {}'.format(lr))
    train(epoch)


