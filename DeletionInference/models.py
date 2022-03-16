import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import sys
from vgg import vgg19_bn
        
class AlexNet(nn.Module):
    def __init__(self, input_size=28, n_channels=1, n_classes=10):
        super(AlexNet, self).__init__()
        if input_size<32:
            add_padding = (32 - input_size) // 2
        else:
            add_padding = 0
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=11, stride=4, padding=5 + add_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.classifier(x), dim=1)
        return x

class smallcnn2(nn.Module):
    def __init__(self, input_size=28, n_channels=1, n_classes=10):
        super(smallcnn2, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 20, 6, 1)
        self.conv_out_size = ((input_size - 4) // 2 - 5) // 2
        self.fc1 = nn.Linear(self.conv_out_size * self.conv_out_size * 20, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.conv_out_size *self.conv_out_size * 20)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class smallcnn(nn.Module):
    def __init__(self, input_size=28, n_channels=1, n_classes=10):
        super(smallcnn, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 128, 5, 1)
        self.conv2 = nn.Conv2d(128, 50, 5, 1)
        self.conv_out_size = ((input_size - 4) // 2 - 4) // 2
        self.fc1 = nn.Linear(self.conv_out_size * self.conv_out_size * 50, 500)
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.conv_out_size *self.conv_out_size * 50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class fcn(nn.Module):
    def __init__(self, input_size=28, n_channels=1, n_classes=10):
        super(fcn, self).__init__()
        self.input_size = input_size * input_size * n_channels
        self.fc = nn.Linear(self.input_size, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class linear(nn.Module):
    def __init__(self, input_size=28, n_channels=1, n_classes=10):
        super(linear, self).__init__()
        self.input_size = input_size * input_size * n_channels
        self.fc = nn.Linear(self.input_size, n_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
        
def load_model(model_name, input_size,n_channels, n_classes):
    if model_name=='fcn':
        return fcn(input_size, n_channels, n_classes)
    elif model_name=='linear':
        return linear(input_size, n_channels, n_classes)
    elif model_name=='cnn':
        return smallcnn(input_size, n_channels, n_classes)
    elif model_name=='cnne':
        return smallcnn2(input_size, n_channels, n_classes)
    elif model_name=='alex':
        return AlexNet(input_size, n_channels, n_classes)
    elif model_name=='vgg':
        return vgg19_bn(n_classes=n_classes)
    else:
        print('Bad Model Name')
        sys.exit(-1)