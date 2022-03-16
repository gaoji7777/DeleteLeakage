import sklearn.datasets
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from types import SimpleNamespace
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import linear_model
from models import load_model
from train import train_model
import json
import argparse

class smalldataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transforms=None):
        self.data = data
        self.label = label
        self.transforms = transforms
        
    def __getitem__(self, index):
        if self.transforms:
            return self.transforms(self.data[index]), self.label[index]
        else:
            return self.data[index], self.label[index]
    def __len__(self):
        return len(self.label)
        
class attackdataset(torch.utils.data.Dataset):
    def __init__(self, data, dclass, label):
        self.data = data
        self.dclass = dclass
        self.label = label
        
    def __getitem__(self, index):
        return self.data[index], self.dclass[index], self.label[index]
            
    def __len__(self):
        return len(self.label)

class attackmodel(nn.Module):
    def __init__(self, n_classes=10):
        super(attackmodel, self).__init__()
        self.fc = nn.Parameter(torch.randn(n_classes, n_classes, 20) * 0.1)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, xtup):
        x, lb = xtup
        x = F.relu(x.unsqueeze(1).bmm(self.fc[lb]).squeeze(1))
        return F.log_softmax(self.fc2(x), dim=1)
        
parser = argparse.ArgumentParser()
parser.add_argument('--m', type=int, default=100, help='number of examples in a batch')
parser.add_argument('--NN', type=int, default=10000, help='total number of examples in the experiment')
parser.add_argument('--data', type=int, default=1, help='which data. 0=CIFAR100, 1=CIFAR10')
parser.add_argument('--model', type=str, default='cnn', help='target model type')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--T', type=int, default=20, help='number of experiment trials')
parser.add_argument('--TT', type=int, default=80, help='number of training epochs')
parser.add_argument('--seed', type=int, default=12, help='random seed')
args =  parser.parse_args()

torch.manual_seed(args.seed)

m = args.m

if args.data == 1:
    cifar_data = datasets.CIFAR10('../data', train=True, download=True)
    cifar_test_data = datasets.CIFAR10('../data', train=False, download=True)
    train_data_tensor = torch.tensor(cifar_data.data.transpose((0,3,1,2))) # HWC -> CHW
    train_data_labels = torch.tensor(cifar_data.targets)    
    test_data_tensor = torch.tensor(cifar_test_data.data.transpose((0,3,1,2)))
    test_data_labels = torch.tensor(cifar_test_data.targets)
    
    NN = args.NN 
    
    dataset_config = json.load(open('configs/%s.json' % 'cifar10', 'r'))
    dataset_config = SimpleNamespace(**dataset_config)

else:
    cifar_data = datasets.CIFAR100('../data', train=True, download=True)
    cifar_test_data = datasets.CIFAR100('../data', train=False, download=True)
    train_data_tensor = torch.tensor(cifar_data.data.transpose((0,3,1,2))) # HWC -> CHW
    train_data_labels = torch.tensor(cifar_data.targets)    
    test_data_tensor = torch.tensor(cifar_test_data.data.transpose((0,3,1,2)))
    test_data_labels = torch.tensor(cifar_test_data.targets)
    
    NN = args.NN 
    
    dataset_config = json.load(open('configs/%s.json' % 'cifar100', 'r'))
    dataset_config = SimpleNamespace(**dataset_config)    


train_data_tensor = train_data_tensor.float()/255.
train_data_tensor = (train_data_tensor - torch.tensor(dataset_config.mean).view(1, -1, 1, 1)) / torch.tensor(dataset_config.std).view(1, -1, 1, 1)
test_data_tensor = test_data_tensor.float()/255.
test_data_tensor = (test_data_tensor - torch.tensor(dataset_config.mean).view(1, -1, 1, 1)) / torch.tensor(dataset_config.std).view(1, -1, 1, 1)


if NN >= 25000:
    # Need to concatenate data
    train_data_tensor = torch.cat((train_data_tensor, test_data_tensor))
    train_data_labels = torch.cat((train_data_labels, test_data_labels))
    
    
N = NN
D = train_data_tensor.size(1)
N_extra = test_data_tensor.size(0)

x_extra = test_data_tensor
yc_extra = test_data_labels
testset = smalldataset(x_extra, yc_extra)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

torch.random.manual_seed(40)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = args.T
succ = 0
succ2 = 0
logi = 0
ss = .0
ss2 = .0
ss3 = .0

def checkid(inds, m=args.m):
    s = 0
    for j in range(m * 2):
        if inds[j]<m:
            s += j
    return m * m - (s - (m - 1) * m / 2)

def sample_data(x, y, slen, return_not_pick=False):
    perm = torch.randperm(x.size(0))
    if return_not_pick:
        return x[perm[:slen]], y[perm[:slen]], x[perm[slen:]], y[perm[slen:]]
    else:
        return x[perm[:slen]], y[perm[:slen]]

def predict(model, x, step=1000):
    model.eval()
    with torch.no_grad():
        s = 0
        outs = []
        while s<x.size(0):
            out = model(x[s:s+step].to(device))
            outs.append(out)
            s += step
        return torch.cat(outs, dim=0)
        
def predicta(model, x, y, step=1000):
    model.eval()
    with torch.no_grad():
        s = 0
        outs = []
        while s<x.size(0):
            out = model((x[s:s+step].to(device), y[s:s+step].to(device)))
            outs.append(out)
            s += step
        return torch.cat(outs, dim=0)
        
def get_data_attack_model(x, y, n_shadow=10, n_examples=args.NN, n_attack=5000):
    xs = []
    ys = []
    lbs = []
    for i in range(n_shadow):
        
        x_train, y_train, x_np, y_np = sample_data(x, y, n_examples, True)
        train_loader = torch.utils.data.DataLoader(smalldataset(x_train, y_train), batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(smalldataset(x_np, y_np), batch_size=128, shuffle=True)
        model_s = load_model(args.model, input_size=dataset_config.input_size, n_channels=dataset_config.n_channels, n_classes=dataset_config.n_classes)
        model_s = model_s.to(device)
        train_model(model_s, device, train_loader, test_loader, lr=args.lr, T1=args.TT)
        
        model_s.eval()
        with torch.no_grad():
            pos_x, pos_y = sample_data(x_train, y_train, n_attack)            
            res_x = predict(model_s, pos_x)
            xs.append(res_x)
            lbs.append(pos_y)
            ys.append(torch.ones(res_x.size(0), dtype=torch.long))
            neg_x, neg_y = sample_data(x_np, y_np, n_attack)            
            res_x = predict(model_s, neg_x)
            xs.append(res_x)
            lbs.append(neg_y)
            ys.append(torch.zeros(res_x.size(0), dtype=torch.long))
            
    xs = torch.cat(xs, dim=0)
    xs = torch.exp(xs)
    lbs = torch.cat(lbs, dim=0)
    ys = torch.cat(ys, dim=0)
    return xs, lbs, ys
        
def train_attack_model(x, lb, y):
    attack_dt = attackdataset(x, lb, y)
    model_a = attackmodel(dataset_config.n_classes)
    model_a = model_a.to(device)
    train_attack_loader = torch.utils.data.DataLoader(attack_dt, batch_size=256, shuffle=True)
    model_a.train()
    optimizer = optim.Adam(model_a.parameters(), lr=0.001)
    for ep in range(500):
        optimizer.zero_grad()
        for xt, lb, yt in train_attack_loader:

            xt, lb, yt = xt.to(device), lb.to(device), yt.to(device)
            optimizer.zero_grad()
            output = model_a((xt,lb))
            loss = F.nll_loss(output, yt)
            loss.backward()
            optimizer.step()
            
        model_a.eval()
        correct = 0
        with torch.no_grad():
            for (xt, lb, yt) in train_attack_loader:
                xt, lb, yt = xt.to(device), lb.to(device), yt.to(device)
                output = model_a((xt, lb))
                pred = output.argmax(dim=1)
                correct += pred.eq(yt.view_as(pred)).sum().item()
        print(100. * correct / len(train_attack_loader.dataset))         
    return model_a

total_res1 = .0
total_res2 = .0
for i in range(T):
    x_train, y_train, x_shadow, y_shadow = sample_data(train_data_tensor, train_data_labels, args.NN, True)
    trainset = smalldataset(x_train, y_train)
    
    x_attack, lb_attack, y_attack = get_data_attack_model(x_shadow, y_shadow, n_shadow=10, n_examples=args.NN)
    model_attack = train_attack_model(x_attack, lb_attack, y_attack)
    
    train_loader = torch.utils.data.DataLoader(smalldataset(x_train, y_train), batch_size=128, shuffle=True)
    model_p = load_model(args.model, input_size=dataset_config.input_size, n_channels=dataset_config.n_channels, n_classes=dataset_config.n_classes)
    model_p = model_p.to(device)
    train_model(model_p, device, train_loader, train_loader, lr=args.lr, T1=args.TT)
    
    corr = 0
    pos_x, pos_y = sample_data(x_train, y_train, 1000)            
    res_x = torch.exp(predict(model_p, pos_x))
    res_ax = predicta(model_attack, res_x, pos_y)
    
    corr += (res_ax.argmax(dim=1) == 1).sum().item()
    neg_x, neg_y = sample_data(x_shadow, y_shadow, 1000)
    
    res_x = torch.exp(predict(model_p, neg_x))
    res_ax = predicta(model_attack, res_x, neg_y)
    corr += (res_ax.argmax(dim=1) == 0).sum().item()
    
    
    x_train_del, y_train_del, x_del, y_del = sample_data(x_train, y_train, args.NN - 100, True)
    x_com, y_com = x_train_del[:100], y_train_del[:100]
    
    train_loader2 = torch.utils.data.DataLoader(smalldataset(x_train_del, y_train_del), batch_size=128, shuffle=True)
    model_p2 = load_model(args.model, input_size=dataset_config.input_size, n_channels=dataset_config.n_channels, n_classes=dataset_config.n_classes)
    model_p2 = model_p2.to(device)
    train_model(model_p2, device, train_loader2, train_loader2, lr=args.lr, T1=args.TT)
    
    res_x = torch.exp(predict(model_p, x_del))
    res_ax = torch.exp(predicta(model_attack, res_x, y_del))
    res_x2 = torch.exp(predict(model_p2, x_del))
    res_ax2 = torch.exp(predicta(model_attack, res_x2, y_del))
    
    res_x_com = torch.exp(predict(model_p, x_com))
    res_ax_com = torch.exp(predicta(model_attack, res_x_com, y_com))
    res_x2_com = torch.exp(predict(model_p2, x_com))
    res_ax2_com = torch.exp(predicta(model_attack, res_x2_com, y_com))
    
    t1 = ((res_ax.argmax(dim=1)==1) * (res_ax2.argmax(dim=1)==0)).sum().item()
    t2 = ((res_ax_com.argmax(dim=1)==1) * (res_ax2_com.argmax(dim=1)==0)).sum().item()
    t1n = ((res_ax.argmax(dim=1)==0) * (res_ax2.argmax(dim=1)==1)).sum().item()
    t2n = ((res_ax_com.argmax(dim=1)==0) * (res_ax2_com.argmax(dim=1)==1)).sum().item()
    
    res1 = (t1/100.) * (1 - 0.5 * t2 / 100.) + (1 - t1/100. - t1n/100.) *  (0.5 * (1 - t2/100. - t2n/100.) + t2n/100.) + (t1n/100.) * 0.5 * (t2n/100.) 
    
    res_cat = torch.cat((res_ax - res_ax2, res_ax_com - res_ax2_com), dim=0)[:, 1]
    inds = res_cat.argsort(descending=True)
    res2 = checkid(inds)

    total_res1 += res1
    total_res2 += res2

f = open('mia.log', 'a')
f.write('cifar %s %d %d %d %.2f%% %.2f%%\n' % (args.model, args.data, args.m, args.NN, total_res1 * 100./T, total_res2 * 100./10000./T))