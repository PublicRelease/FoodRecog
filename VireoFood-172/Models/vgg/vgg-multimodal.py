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

# —Path settings———————————————————————————————————————————————————————————————————————————————————————————————————————
root_path = '/mnt/FoodRecog/'  # /home/lily/Desktop/food/ /Users/lei/PycharmProjects/FoodRecog/ /mnt/FoodRecog/
image_folder = 'ready_chinese_food'  # scaled_images ready_chinese_food
image_path = os.path.join(root_path, image_folder, '/')

file_path = os.path.join(root_path, 'SplitAndIngreLabel/')
ingredient_path = os.path.join(file_path, 'IngreLabel.txt')

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

        elif test_data:
            with io.open(test_data_path, encoding='utf-8') as file:
                path_to_images = file.read().split('\n')
            labels = matio.loadmat(file_path + 'test_label.mat')['test_label'][0, :]
            ingredients = matio.loadmat(file_path + 'wordIndicator_test.mat')['wordIndicator_test']

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
    def __init__(self):
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


# encoder network for textual channel
class encoder_t(nn.Module):
    def __init__(self):
        super(encoder_t, self).__init__()

        batchNorm_momentum = 0.1

        self.nn1 = nn.Linear(309, 4096)
        self.bn1 = nn.BatchNorm1d(4096, momentum=batchNorm_momentum)
        self.nn2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096, momentum=batchNorm_momentum)

    def forward(self, y):
        # compute latent vectors
        y = F.relu(self.bn1(self.nn1(y)))
        y_latent = F.relu(self.bn2(self.nn2(y)))
        return y_latent


# decoder network for textual channel
class decoder_t(nn.Module):
    def __init__(self):
        super(decoder_t, self).__init__()

        batchNorm_momentum = 0.1

        self.nn2d = nn.Linear(4096, 4096)
        self.bn2d = nn.BatchNorm1d(4096, momentum=batchNorm_momentum)

        self.nn1d = nn.Linear(4096, 309)

    def forward(self, y):  # x:image, y:ingredient
        # compute reconstructed vectors
        y = F.relu(self.bn2d(self.nn2d(y)))
        y_recon = self.nn1d(y)

        return y_recon


# entire model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # network for image channel
        self.encoder = vgg19_bn(pretrained=True)
        self.decoder = SegNet()

        # network for ingredient channel
        # encoder
        self.encoder_t = encoder_t()
        self.decoder_t = decoder_t()

        # classifier
        self.classifier = nn.Linear(blk_len, 172)

    def forward(self, x, y):  # x:image, y:ingredient
        # compute image latent vectors & recons
        x_latent = self.encoder(x)
        x_recon = self.decoder(x_latent)

        # compute predicts
        predicts = self.classifier(x_latent[:, 0:blk_len])

        # compute ingredient vectors
        y_latent = self.encoder_t(y)
        predicts_t = self.classifier(y_latent[:, 0:blk_len])
        y_recon = self.decoder_t(y_latent)

        return predicts, predicts_t, x_recon, y_recon, x_latent[:, 0:blk_len], y_latent[:, 0:blk_len]


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

# Model
model = MyModel()

if CUDA:
    model = nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.module.parameters(), weight_decay=1e-3,
                       lr=learning_rate)  # .module.parameters(), lr=learning_rate)

# ------------------------------------------------------------------

# Loss
criterion = nn.CrossEntropyLoss()


# l1_norm = nn.L1Loss()


def loss_function(predicts_V, predicts_T, labels, data, ingredients, x_recon, y_recon, x_latent, y_latent):
    # image channel loss
    CE_V = criterion(predicts_V, labels - 1) * 20
    CE_T = criterion(predicts_T, labels - 1)

    # ingredient channel loss
    RE_V = (torch.sum((data - x_recon) ** 2)) * (1e-5)
    RE_T = (torch.sum((ingredients - y_recon) ** 2)) * (1e-2)

    # l2 norm for feature alignment
    AE = torch.sum((x_latent - y_latent) ** 2) * (1e-3)

    return CE_V, CE_T, RE_V, RE_T, AE


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

    for batch_idx, (data, labels, ingredients) in enumerate(train_loader):
        # ---------------------------------------------------------------------------------------------------------------------------------
        # for effective code debugging
        # if batch_idx == 2:
        #    break
        # print('batch %',batch_idx)
        # ---------------------------------------------------------------------------------------------------------------------------------

        start_time = time.time()
        data = Variable(data)
        ingredients = Variable(ingredients)
        if CUDA:
            data = data.cuda()
            labels = labels.cuda()
            ingredients = ingredients.cuda()

        # obtain output from model
        predicts_V, predicts_T, x_recon, y_recon, x_latent, y_latent = model(data, ingredients)

        # loss
        CE_V, CE_T, RE_V, RE_T, AE = loss_function(predicts_V, predicts_T, labels, data, ingredients, x_recon,
                                                   y_recon, x_latent, y_latent)

        # optim for myModel with generator
        optimizer.zero_grad()
        loss = CE_V + CE_T + RE_V + RE_T + AE
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
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | RE_V: {:.4f} | RE_T: {:.4f} | AE: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader),
                    loss.data, CE_V.data, CE_T.data, RE_V.data, RE_T.data, AE.data,
                    top1_accuracy_cur_V, top5_accuracy_cur_V,
                    round((time.time() - start_time), 4),
                    round((time.time() - total_time), 4)))

            with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
                # print('write in-batch loss at epoch {} | batch {}'.format(epoch,batch_idx))
                file.write('%f\n' % (train_loss))

        elif batch_idx % LOG_INTERVAL == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | RE_V: {:.4f} | RE_T: {:.4f} | AE: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader),
                    loss.data, CE_V.data, CE_T.data, RE_V.data, RE_T.data, AE.data,
                    top1_accuracy_cur_V,
                    top5_accuracy_cur_V,
                    round((time.time() - start_time) * LOG_INTERVAL, 4),
                    round((time.time() - total_time), 4)))

        # records current progress for tracking purpose
        with io.open(result_path + 'model_batch_train_loss.txt', 'w', encoding='utf-8') as file:
            file.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | CE_T: {:.4f} | RE_V: {:.4f} | RE_T: {:.4f} | AE: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader),
                    loss.data, CE_V.data, CE_T.data, RE_V.data, RE_T.data, AE.data,
                    top1_accuracy_cur_V,
                    top5_accuracy_cur_V,
                    round((time.time() - start_time) * LOG_INTERVAL, 4),
                    round((time.time() - total_time), 4))
            )

    print(
        '====> Epoch: {} | Average loss: {:.4f} | Average Top1_Accuracy_V:{} | Average Top5_Accuracy_V:{} | Time:{}'.format(
            epoch, train_loss / len(train_loader), top1_accuracy_total_V / len(train_loader.dataset),
                   top5_accuracy_total_V / len(train_loader.dataset), round((time.time() - total_time), 4)))

    with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
        # print('write in-epoch loss at epoch {} | batch {}'.format(epoch,batch_idx))
        file.write('%f\n' % (train_loss / len(train_loader)))

    # save current model
    if epoch > 3:
        torch.save(model.state_dict(), result_path + 'model' + str(epoch) + '.pt')


def lr_scheduler(optimizer, init_lr, epoch, lr_decay_iter):
    if epoch % lr_decay_iter:
        return init_lr

    lr = init_lr * 0.1  # drop to 0.1*init_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


decay = 4
EPOCHS = decay * 2 + 1

for epoch in range(1, EPOCHS + 1):
    learning_rate = lr_scheduler(optimizer, learning_rate, epoch, decay)
    print(learning_rate)
    train(epoch)
# test(epoch)

# the_model = TheModelClass(*args, **kwargs)
# the_model.load_state_dict(torch.load(PATH))
# On the last epoch, test performance on the 32 test samples
# tester()

