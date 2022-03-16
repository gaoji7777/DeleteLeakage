import torch
import torch.nn as nn
import torch.nn.functional as F

class linear(nn.Module):
    def __init__(self, input_size=(1, 28, 28), n_classes = 10):
        super(linear, self).__init__()
        self.input_size = input_size[0] * input_size[1] * input_size[2]
        self.fc = nn.Linear(self.input_size, n_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
        
class twolayer(nn.Module):
    def __init__(self, input_size=(1, 28, 28), n_classes = 10, hidden_size=500):
        super(twolayer, self).__init__()
        self.input_size = input_size[0] * input_size[1] * input_size[2]
        self.fc = nn.Linear(self.input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
class smallcnn(nn.Module):
    def __init__(self, input_size=(1, 28, 28), n_classes=10):
        n_channels = input_size[0]
        super(smallcnn, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 128, 5, 1)
        self.conv2 = nn.Conv2d(128, 50, 5, 1)
        self.conv_out_size = ((input_size[1] - 4) // 2 - 4) // 2
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
        
class nearestneighbor(nn.Module):
    def __init__(self, x, y, input_size=(1, 28, 28), n_classes = 10):
        super(nearestneighbor, self).__init__()
        self.x = x.view(x.size(0), -1)
        self.y = y

    def forward(self, x_test, k=1):
        x_test = x_test.view(x_test.size(0), -1)
        dist_full = x_test.mm(self.x.transpose(0, 1)) + (1 - x_test).mm((1 - self.x).transpose(0, 1)) 
        yks = self.y[dist_full.topk(k, largest=True, dim=1)[1]]
        out = []
        for yk in yks:
            ret, count = yk.unique(return_counts=True)
            out.append(ret[count.argmax()])
        return torch.stack(out)