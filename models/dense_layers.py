import torch.nn as nn
import torch.nn.functional as F
import torch

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.ml1 = nn.Linear(2,1,bias=False)

    def forward(self, x):
        x = self.ml1(x)
        return x
    



class SmallANN(nn.Module):
    def __init__(self, device):
        super(SmallANN, self).__init__()
        self.ml1 = nn.Linear(2, 1000)
        self.ml2 = nn.Linear(1000, 1)


    def forward(self, x):
        x = self.ml1(x)
        x = F.relu(x)
        x = self.ml2(x)
        x = F.sigmoid(x)


        return x

    def get_layers(self):
        return [self.ml1, self.ml2]

    def print_weights(self):
        print('FC 1: ', self.ml1.weight.sum().item(), torch.abs(self.ml1.weight).sum().item())
        print('FC 2: ', self.ml2.weight.sum().item(), torch.abs(self.ml2.weight).sum().item())



class DenseANN(nn.Module):
    def __init__(self, device):
        super(DenseANN, self).__init__()
        self.ml1 = nn.Linear(784, 1000)
        self.ml2 = nn.Linear(1000, 1000)
        self.ml3 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.ml1(x)
        x = F.relu(x)
        x = self.ml2(x)
        x = F.relu(x)
        x = self.ml3(x)

        return x

    def get_layers(self):
        return [self.ml1, self.ml2, self.ml3]

    def print_weights(self):
        print('FC 1: ', self.ml1.weight.sum().item(), torch.abs(self.ml1.weight).sum().item())
        print('FC 2: ', self.ml2.weight.sum().item(), torch.abs(self.ml2.weight).sum().item())
        print('FC 3: ', self.ml3.weight.sum().item(), torch.abs(self.ml3.weight).sum().item())


class BigDenseANN(nn.Module):
    def __init__(self, device):
        super(BigDenseANN, self).__init__()
        self.ml1 = nn.Linear(784, 800)
        self.ml2 = nn.Linear(800, 200)
        self.ml3 = nn.Linear(200, 47)

    def forward(self, x):
        x = self.ml1(x)
        x = F.relu(x)
        x = self.ml2(x)
        x = F.relu(x)
        x = self.ml3(x)

        return x

    def get_layers(self):
        return [self.ml1, self.ml2, self.ml3]

    def print_weights(self):
        print('FC 1: ', self.ml1.weight.sum().item(), torch.abs(self.ml1.weight).sum().item())
        print('FC 2: ', self.ml2.weight.sum().item(), torch.abs(self.ml2.weight).sum().item())
        print('FC 3: ', self.ml3.weight.sum().item(), torch.abs(self.ml3.weight).sum().item())


class Dense4CNN(nn.Module):
    expansion = 1

    def __init__(self, device=None):
        super(Dense4CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same', bias=False, device=device)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False, device=device)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same', bias=False, device=device)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False, device=device)

        self.dense1 = nn.Linear(6272, 256, device=device)
        self.dense2 = nn.Linear(256, 256, device=device)
        self.dense3 = nn.Linear(256, 10, device=device)

    def forward(self, x, ths=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
class Dense6CNN(nn.Module):
    expansion = 1

    def __init__(self, device=None):
        super(Dense6CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same')
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same')

        self.dense1 = nn.Linear(4096, 256)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class Dense8CNN(nn.Module):
    expansion = 1

    def __init__(self, device=None):
        super(Dense8CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same')
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same')
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same')
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same')

        self.dense1 = nn.Linear(2048, 256)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(F.relu(self.conv8(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class Dense10CNN(nn.Module):
    expansion = 1

    def __init__(self, device=None):
        super(Dense10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same')
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same')
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same')
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same')
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding='same')
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding='same')

        self.dense1 = nn.Linear(1024, 256)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(F.relu(self.conv8(x)), kernel_size=2, stride=2)
        x = F.relu(self.conv9(x))
        x = F.max_pool2d(F.relu(self.conv10(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x