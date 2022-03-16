import numpy as np
from tqdm import tqdm
from loaddata import load
import torch
import argparse
from loadmodels import nearestneighbor
from torchvision.utils import save_image
import time
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('--seed', type=int, default=120, help='random seed')
parser.add_argument('--K', type=int, default=1, help='K in K-NN')
parser.add_argument('--data', type=str, default='binary_omniglot', help='dataset name')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

X_train, X_test, y_train, y_test = load(args.data)

X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)
perm = torch.randperm(X_train.size(0))
X_train = X_train[perm]
y_train = y_train[perm]
perm = torch.randperm(X_test.size(0))
X_test = X_test[perm]
y_test = y_test[perm]

fig_size = X_train.size(1)
print('Start experiment')
learner = nearestneighbor
kwargs = {}

def del_inf(learner, kwargs, X_train, X_test, y_train, y_test, K=5):
    base_model = learner(X_train, y_train)
    y_test_pred = base_model(X_test, k=K)
    exps = []
    exp_orig = []

    for l in range(100):
        exps.append((X_test[(y_test_pred == y_train[l]).nonzero().view(-1)].mean(dim=0) >= 0.5).float())
        exp_orig.append(X_train[l])
        
    exps = torch.cat(exps, dim=0)
    exp_orig = torch.cat(exp_orig, dim=0)
    stackp = torch.cat((exps[:11].view(-1, 1, 105, 105), exp_orig[:11].view(-1, 1, 105, 105)), dim=0)
    for i in range(1, (exp_orig.size(0) - 1 )//11 + 1):
        stackp = torch.cat((stackp, exps[i * 11 :(i+1) * 11].view(-1, 1, 105, 105), exp_orig[i * 11 :(i + 1) * 11].view(-1, 1, 105, 105)), dim=0)

    fs = min(X_train.shape[0], 100)
        
    succ = 0
    succ2 = 0
    
    X_preds = []
    X_base = []
    exps_p = []
    
    thres = X_test.mean(dim=0)
    for count in tqdm(range(fs)):
        del_model = learner(torch.cat((X_train[:count], X_train[count+1:]), dim=0), torch.cat((y_train[:count], y_train[count+1:]), dim=0))
        del_sample = X_train[count]
        y_test_pred_del = del_model(X_test, k=K)
        notmatch = (y_test_pred_del != y_test_pred)
        if notmatch.sum()!=0:
            X_notmatch = X_test[notmatch.nonzero().view(-1)]
            X_pred = (X_notmatch.mean(dim=0) >= 0.5).float()
            exps_p.append(exps[count])
            X_preds.append(X_pred)
            X_base.append(del_sample)
            y_ll = base_model(X_pred.unsqueeze(0), k=K)
            y_del = base_model(del_sample.unsqueeze(0), k=K)
            if y_ll[0].item() == y_del[0].item():
                succ2 += 1
            succ += 1
            
    X_preds = torch.cat(X_preds, dim=0)
    X_base = torch.cat(X_base, dim=0)
    exps_p = torch.cat(exps_p, dim=0)
    
    row_size = 11
    stack = torch.cat((exps_p[:row_size].view(-1, 1, 105, 105), X_preds[:row_size].view(-1, 1, 105, 105),  X_base[:row_size].view(-1, 1, 105, 105)), dim=0)
    
    for i in range(1, (X_base.size(0) - 1 )//row_size + 1):
        if i != (X_base.size(0) - 1 ) // row_size:
            stack = torch.cat((stack, exps_p[i * row_size :(i+1) * row_size].view(-1, 1, 105, 105), X_preds[i * row_size :(i+1) * row_size].view(-1, 1, 105, 105), X_base[i * row_size :(i + 1) * row_size].view(-1, 1, 105, 105)), dim=0)
        else:
            pad = row_size - ((len(exps_p)) - i * row_size)
            stack =  torch.cat((stack, exps_p[i * row_size :(i+1) * row_size].view(-1, 1, 105, 105), torch.zeros(pad, 1, 105, 105), X_preds[i * row_size :(i+1) * row_size].view(-1, 1, 105, 105), torch.zeros(pad, 1, 105, 105), X_base[i * row_size :(i + 1) * row_size].view(-1, 1, 105, 105)), dim=0)

    save_image(stack, "X_pred_omniglot_new_knn_%d_%d.png" % (K, int(time.time())), nrow=row_size, normalize=True)

    print('Del Inf Succ: %.2f%%' % (succ * 100./fs))
    print('Del Inf Succ2: %.2f%%' % (succ2 * 100./fs))

del_inf(learner, kwargs, X_train, X_test, y_train, y_test, K=args.K)
