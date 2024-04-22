import copy
import torch.optim
import torch
import torch.nn as nn
import numpy as np

from ..models.dense_layers import *
from ..models.lenet import *
from ..models.resnet import *
from ..models.vgg import *


def get_model(params, device):
    model, optimizer = None, None

    if params.get('model').get('id') == 'conv4':
        model = Dense4CNN()
    elif params.get('model').get('id') == 'conv6':
        model = Dense6CNN()
    elif params.get('model').get('id') == 'lenet':
        model = LeNet()
    elif params.get('model').get('id') == 'conv8':
        model = Dense8CNN()
    elif params.get('model').get('id') == 'conv10':
        model = Dense10CNN()
    elif params.get('model').get('id') == 'resnet18':
        model = ResNet18(num_classes=params.get('data').get('num_classes'))
    elif "VGG" in params.get("model").get("id"):
        model = VGG(architecture_name=params.get("model").get("id"),
                    dataset_name=params.get("data").get("dataset"))
    model.to(device)

    if params.get('model').get('optimizer').get('optim') == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     params.get('model').get('optimizer').get("local_lr"))
    elif params.get('model').get('optimizer').get('optim') == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                     params.get('model').get('optimizer').get("local_lr"),
                                    momentum=params.get('model').get('optimizer').get("mu"),
                                    weight_decay=params.get('model').get('optimizer').get("weight_decay"))
    else:
        print('Not Implemented Error: {} not implemented' .format(params.get('model').get('optimizer').get('optim')))


    return model, optimizer

def get_gradients(model):
    grads = {}
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            grads[k] = copy.deepcopy(m.weight.grad.data).cpu().numpy()
    return grads

def find_cosine_dist(curr_model, prev_grads):
    dot_prod = 0
    curr_norm = 0
    prev_norm = 0
    for k, m in enumerate(curr_model.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m_numpy = copy.deepcopy(m.weight.grad.data).cpu().numpy()
            dot_prod += np.sum(m_numpy * prev_grads[k])
            curr_norm += np.square(np.linalg.norm(m_numpy))
            prev_norm += np.square(np.linalg.norm(prev_grads[k]))
    return dot_prod / (np.sqrt(curr_norm * prev_norm))

