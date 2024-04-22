"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

class WideBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, device=None):
        super(WideBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(planes, device=device)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn2 = nn.BatchNorm2d(planes, device=device)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, device=device),
                nn.BatchNorm2d(self.expansion*planes, device=device)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WideBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, device=None):
        super(WideBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(planes, device=device)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, device=device)
        self.bn2 = nn.BatchNorm2d(planes, device=device)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False, device=device)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, device=device)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, device=device),
                nn.BatchNorm2d(self.expansion*planes, device=device)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WideResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, device=None):
        super(WideResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(64, device=device)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, device=device)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, device=device)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, device=device)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, device=device)
        self.linear = nn.Linear(512*block.expansion, num_classes, device=device)

    def _make_layer(self, block, planes, num_blocks, stride, device=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, device=device))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # return out
        # date : 04 Apr, adding this to avoid backprop errors
        # return F.log_softmax(out, dim=1)
        return F.sigmoid(out)

def ResNet18(num_classes=10, device=None):
    return WideResNet(WideBasicBlock, [2,2,2,2], num_classes=num_classes, device=device)

def ResNet34(num_classes=10, device=None):
    return WideResNet(WideBasicBlock, [3,4,6,3], num_classes=num_classes, device=device)

def ResNet50(num_classes=10, device=None):
    return WideResNet(WideBottleneck, [3,4,6,3], num_classes=num_classes, device=device)

def ResNet101(num_classes=10, device=None):
    return WideResNet(WideBottleneck, [3,4,23,3], num_classes=num_classes, device=device)

def ResNet152(num_classes=10, device=None):
    return WideResNet(WideBottleneck, [3,8,36,3], num_classes=num_classes, device=device)
