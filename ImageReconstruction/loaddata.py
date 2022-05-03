import torch
from torchvision import datasets, transforms
import sys

def load_omniglot(dataset_name, omniglot_lang=True):
    dataset = datasets.Omniglot(
        root="../data", download=True, transform=transforms.ToTensor()
    )
    Xs = []
    ys = []
    
    for (image, label) in dataset:
        Xs.append(image)
        ys.append(label)
    Xs = torch.cat(Xs, dim=0)
    ys = torch.LongTensor(ys)
    
    
    y_max = ys.max()
    
    perm = torch.randperm(ys.size(0))
    Xs = Xs[perm]
    ys = ys[perm]
    
    n_train = 10
    n_test = 10
    
    assert n_train + n_test <= 20
    
    train_lbs = []
    test_lbs = []
    chars = [char.split('/')[0] for char in dataset._characters]
    chars_u = list(set(chars))
    dt = {}
    for i, char in enumerate(chars_u):
        dt[char] = i 
    y_ch = torch.LongTensor([dt[chars[ys[yi]]] for yi in range(len(ys))])
    
    if omniglot_lang:
        ys = y_ch
        y_max = ys.max()
        n_train = 140
        n_test = 140
    
    for yi in range(y_max + 1):
        train_lbs.append((ys == yi).nonzero()[:n_train])
        test_lbs.append((ys == yi).nonzero()[n_train: n_test + n_train])
    
    train_lbs = torch.cat(train_lbs)
    test_lbs = torch.cat(test_lbs)
    
    X_train = Xs[train_lbs]
    X_test = Xs[test_lbs]
    y_train = ys[train_lbs]
    y_test = ys[test_lbs]
    return X_train - 0.5, X_test - 0.5, y_train, y_test
    
def binarize(X_train, X_test, y_train, y_test):
    X_train = (X_train >= 0).float()
    X_test = (X_test >= 0).float()
    return X_train, X_test, y_train, y_test
    
def load(dataset_name, **kwargs):
    if dataset_name in ['binary_omniglot']:
        original_dataset_name = dataset_name.split('_')[1]
        return binarize(*load_omniglot(original_dataset_name, **kwargs))
    else:
        sys.exit("Not implemented choice of data")
    
    