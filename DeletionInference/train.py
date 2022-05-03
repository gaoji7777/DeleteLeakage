import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


def train_model(model, device, train_loader, test_loader, lr=0.001, T1=30, optimizer='sgd'):
    model.train()
    if optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    for ep in range(T1):
        optimizer.zero_grad()
        for xt, yt in train_loader:

            xt, yt = xt.to(device), yt.to(device)
            optimizer.zero_grad()
            output = model(xt)
            loss = F.nll_loss(output, yt)
            loss.backward()
            optimizer.step()
            
        if test_loader:
            evaluate(test_loader, model, device)
        
def train_sgd(model, device, x, y, test_loader, lr=0.001, T1=1, T2=1, batch_size=1):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
    for _ in range(T1):
        optimizer.zero_grad()
        for i in tqdm(range( (x.size(0) - 1) // batch_size + 1)):
            for _ in range(T2):
                xt, yt = x[i * batch_size : (i + 1) * batch_size].to(device), y[i * batch_size : (i + 1) * batch_size].to(device)
                optimizer.zero_grad()
                output = model(xt)
                loss = F.nll_loss(output, yt)
                loss.backward()
                optimizer.step()
        evaluate(test_loader, model, device)

def evaluate(eva_loader, model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for (data, label) in eva_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(label.view_as(pred)).sum().item()
    return (100. * correct / len(eva_loader.dataset))

def make_pred(model, device, x, batch_size=128):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range( (x.size(0) - 1) // batch_size + 1):
            xt = x[i * batch_size : (i + 1) * batch_size].to(device)
            output = model(xt)
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    return outputs

def make_pred_sk(model, device, x, batch_size=128):
    outputs = []
    with torch.no_grad():
        for i in range( (x.size(0) - 1) // batch_size + 1):
            xt = x[i * batch_size : (i + 1) * batch_size]
            output = torch.FloatTensor(model.predict_proba(xt))
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    return outputs
    
def examine(model, device, x, y, batch_size=128):
    model.eval()
    correct = 0
    losses = []
    labels = []
    corrects = []
    with torch.no_grad():
        for i in range( (x.size(0) - 1) // batch_size + 1):
            xt, yt = x[i * batch_size : (i + 1) * batch_size].to(device), y[i * batch_size : (i + 1) * batch_size].to(device)
            output = model(xt)
            losses.append(F.nll_loss(output, yt, reduction='none'))
            corrects.append(output.cpu().argmax(dim=1) == yt.cpu())
            labels.append(yt.cpu())
    losses = torch.cat(losses, dim=0)
    corrects = torch.cat(corrects, dim=0)
    labels = torch.cat(labels, dim=0)
    return losses, corrects, labels
    
def batchify(t, k):
    return t.view(-1, k)
    

def examine_k(model, device, x, y, k=50, batch_size=128):
    losses, corrects, labels = examine(model, device, x, y)
    losses = batchify(losses, k)
    corrects = batchify(corrects, k)
    labels = batchify(labels, k)
    losses = losses.mean(dim=-1)
    corrects = corrects.sum(dim=-1)
    # labels = labels.sum(dim=-1)
    return losses, corrects, labels