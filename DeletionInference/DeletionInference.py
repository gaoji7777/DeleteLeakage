import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from types import SimpleNamespace
from models import load_model
from train import train_model
import json
import argparse
torch.random.manual_seed(12)

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

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=int, default=100, help='number of examples in a batch')
parser.add_argument('--NN', type=int, default=10000, help='total number of examples in the experiment')
parser.add_argument('--data', type=int, default=1, help='which data. 0=CIFAR100, 1=CIFAR10')
parser.add_argument('--label', type=int, default=1, help='infer with/without label. 0=without label, 1=with label')
parser.add_argument('--model', type=str, default='cnn', help='target model type')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--T', type=int, default=20, help='number of experimental trails')
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


N = NN
D = train_data_tensor.size(1)
N_extra = test_data_tensor.size(0)

x_extra = test_data_tensor
yc_extra = test_data_labels
testset = smalldataset(x_extra, yc_extra)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

torch.random.manual_seed(25)

T = args.T
succ = 0
succ2 = 0

def checkid(inds, m=args.m):
    s = 0
    for j in range(m * 2):
        if inds[j]<m:
            s += j
    return m * m - (s - (m - 1) * m / 2)

for i in range(T):
    perm_t = torch.randperm(train_data_tensor.size(0))
    train_data_tensor_t = train_data_tensor[perm_t]
    train_data_labels_t = train_data_labels[perm_t]
    x = train_data_tensor_t[:NN]
    yc = train_data_labels_t[:NN]

    trainset = smalldataset(x, yc)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_p = load_model(args.model, input_size=dataset_config.input_size, n_channels=dataset_config.n_channels, n_classes=dataset_config.n_classes)
    model_p = model_p.to(device)

    train_model(model_p, device, train_loader, test_loader, lr=args.lr, T1=args.TT)
    
    model_p.eval()
    
    perm = torch.randperm(N)
    remove_ind = perm[:m]
    rest_ind = perm[m:]
    compare_ind = perm[m : m * 2]
    x_rem = x[rest_ind]
    yc_rem = yc[rest_ind]
    extra_id = torch.randperm(N_extra)[:m]
    x_extra_t = x_extra[extra_id]
    yc_extra_t = yc_extra[extra_id]
    
    remset = smalldataset(x_rem, yc_rem)
    rem_loader = torch.utils.data.DataLoader(remset, batch_size=128, shuffle=True)
    
    model_r = load_model(args.model, input_size=dataset_config.input_size, n_channels=dataset_config.n_channels, n_classes=dataset_config.n_classes)
    model_r = model_r.to(device)
    train_model(model_r, device, rem_loader, test_loader, lr=args.lr, T1=args.TT)
    
    model_r.eval()
    test_ind = torch.cat((remove_ind, compare_ind), dim=0)
    if args.label:    
        with torch.no_grad():
            res_before = F.nll_loss(model_p(x[test_ind].to(device)), yc[test_ind].view(-1).to(device)  , reduction='none').cpu()
            res_after = F.nll_loss(model_r(x[test_ind].to(device)), yc[test_ind].view(-1).to(device) , reduction='none').cpu()
            res_before_extra = F.nll_loss(model_p(x_extra_t.to(device)), yc_extra_t.view(-1).to(device) , reduction='none').cpu()
            res_after_extra = F.nll_loss(model_r(x_extra_t.to(device)), yc_extra_t.view(-1).to(device) , reduction='none').cpu()

        inds = (res_after - res_before).view(-1).argsort(descending=True)
        succ += checkid(inds)
        diff_extra = torch.cat(((res_after - res_before)[:m], res_after_extra - res_before_extra), dim=0)
        inds2 = diff_extra.view(-1).argsort(descending=True)
        succ2 += checkid(inds2)
    else:
        with torch.no_grad():
            res_before = torch.exp(model_p(x[test_ind].to(device))).cpu()
            res_after = torch.exp(model_r(x[test_ind].to(device))).cpu()
            res_before_extra= torch.exp(model_p(x_extra_t.to(device))).cpu()
            res_after_extra = torch.exp(model_r(x_extra_t.to(device))).cpu()
        inds = (res_after - res_before).norm(dim=-1).view(-1).argsort(descending=True)
        succ += checkid(inds)
        diff_extra = torch.cat(((res_after - res_before).norm(dim=-1)[:m].view(-1).abs(), (res_after_extra - res_before_extra).norm(dim=-1).view(-1).abs()), dim=0)
        inds2 = diff_extra.view(-1).argsort(descending=True)
        succ2 += checkid(inds2)

f = open('deletion.log', 'a')
f.write('%s %d %d %d %.2f%%\n' % (args.model, args.data, args.label, args.NN, (succ + .0) / (T * m * m) * 100))