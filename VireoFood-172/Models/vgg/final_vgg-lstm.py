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

            with io.open(validation_data_path, encoding='utf-8') as file:
                path_to_images1 = file.read().split('\n')
            labels1 = matio.loadmat(file_path + 'validation_label.mat')['validation_label']

            path_to_images = path_to_images + path_to_images1
            labels = np.concatenate([labels, labels1], 1)[0, :]

        self.path_to_images = path_to_images
        self.labels = labels

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

        return img, label

    def __len__(self):
        return len(self.path_to_images)


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


set_gpu_encoder = 3
set_gpu_others = 1


# entire model
class MyModel(nn.Module):
    def __init__(self, num_key_ingre=5, max_seq=30):
        super(MyModel, self).__init__()
        # network for image channel
        self.encoder = vgg19_bn().cuda(set_gpu_encoder)
        self.vgg_map2vec = nn.Linear(512 * ((image_size[0] // (2 ** 5)) ** 2), 4096).cuda(set_gpu_others)
        self.vgg_linear = nn.Linear(4096, 4096).cuda(set_gpu_others)

        # classifier
        self.classifier_v = nn.Linear(blk_len, 172).cuda(set_gpu_others)

        # domain transfer
        self.trans_img2l = nn.Linear(blk_len, blk_len).cuda(set_gpu_others)

        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()

        self._initialize_weights()

    def forward(self, x):  # x:image, y:ingredient
        # compute image latent vectors & recons
        x_latent_maps = self.encoder(x)
        x_latent = self.get_latent(x_latent_maps.cuda(set_gpu_others))

        # compute v t predicts in domain adapted space
        predicts = self.get_predicts_with_align(x_latent[:, 0:blk_len])

        return predicts.cuda(set_gpu_others)

    def get_latent(self, x_latent_maps):
        x_latent = x_latent_maps.view(x_latent_maps.size(0), -1)
        x_latent = self.dropout(self.relu(self.vgg_map2vec(x_latent)))
        x_latent = self.dropout(self.relu(self.vgg_linear(x_latent)))
        return x_latent

    def get_predicts_with_align(self, x_latent):
        # compute features in the transferred latent domain
        x_latent2l = self.trans_img2l(x_latent)
        predicts = self.classifier_v(x_latent2l)
        return predicts

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


# —Model training & testing———————————————————————————————————————————————————————————————————————————————————————————————————————
def get_updateModel():
    model_dict = model.state_dict()

    # update image channel
    pretrained_dict = torch.load(file_path + 'model9-words.pt', map_location='cpu')
    extracted_dict_img = {k[7:]: v for k, v in pretrained_dict.items() if
                          k.startswith('module.encoder.') and not k.startswith('module.encoder.classifier')}

    #    # update lstm channel
    #    pretrained_dict = torch.load(file_path + 'finalModel_vgg-lstm.pt', map_location='cpu')
    #    extracted_dict_lstm = {k: v for k, v in pretrained_dict.items() if
    #                           k.startswith('encoder_t.')
    #                           or k.startswith('decoder_t.')
    #                           or k.startswith('classifier_')
    #                           or k.startswith('trans_')
    #                           }

    model_dict.update(extracted_dict_img)
    #    model_dict.update(extracted_dict_lstm)
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


def loss_function(predicts_V, labels):
    # image channel loss
    CE_V = criterion(predicts_V, labels - 1) * 20

    return CE_V


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

    for batch_idx, (data, labels) in enumerate(train_loader):
        # ---------------------------------------------------------------------------------------------------------------------------------
        # for effective code debugging
        # if batch_idx < len(train_loader)-20:
        #    print('skip batch {}'.format(batch_idx))
        #    continue
        # print('batch %',batch_idx)
        # ---------------------------------------------------------------------------------------------------------------------------------

        start_time = time.time()
        data = Variable(data)
        if CUDA:
            data = data.cuda(set_gpu_encoder)
            labels = labels.cuda(set_gpu_others)

        # obtain output from model
        predicts_V = model(data)

        # loss
        CE_V = loss_function(predicts_V, labels)

        # optim for myModel with generator
        optimizer.zero_grad()
        loss = CE_V
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
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.data,
                    CE_V.data,
                    top1_accuracy_cur_V, top5_accuracy_cur_V,
                    round((time.time() - start_time), 4),
                    round((time.time() - total_time), 4)))

            with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
                # print('write in-batch loss at epoch {} | batch {}'.format(epoch,batch_idx))
                file.write('%f\n' % (train_loss))

        elif batch_idx % LOG_INTERVAL == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.data,
                    CE_V.data,
                    top1_accuracy_cur_V, top5_accuracy_cur_V,
                           round((time.time() - start_time), 4) * LOG_INTERVAL,
                    round((time.time() - total_time), 4)))

        # records current progress for tracking purpose
        with io.open(result_path + 'model_batch_train_loss.txt', 'w', encoding='utf-8') as file:
            file.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | CE_V: {:.4f} | Top1_Accuracy_V:{} | Top5_Accuracy_V:{} | Time:{} | Total_Time:{}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.data,
                    CE_V.data,
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
    if lr > 0:
        optimizer.param_groups[0]['lr'] = lr

    return lr


decay = 4
EPOCHS = decay * 3 + 1

for epoch in range(1, EPOCHS + 1):
    learning_rate = lr_scheduler(optimizer, learning_rate, epoch, decay)
    print(learning_rate)
    train(epoch)




