# Comparing deletion inference with membership inference on large models/datasets

In this experiment, we run membership inference and deletion inference on large deep neural networks. 

- This experiment can take a few hours to finish, depending on each device.
- The result is used in the section "Attacking large models and datasets", specifically the data of Fig.1 and Fig.2, in the paper.


## Parameters for membership inference: 
```
python MIA.py [-h] [--m M] [--NN NN] [--data DATA] [--model MODEL] [--lr LR] [--T T] [--TT TT] [--seed SEED]

optional arguments:
  -h, --help     show this help message and exit
  --m M          number of examples in a batch
  --NN NN        total number of examples in the experiment
  --data DATA    which data. 0=CIFAR100, 1=CIFAR10
  --model MODEL  target model type
  --lr LR        learning rate
  --T T          number of experiment trials
  --TT TT        number of training epochs
  --seed SEED    random seed
```

- For Fig.1, set '--model cnn'.
- For Fig.2, set '--model vgg'.
- Output: output is written in one line of the file 'mia.log', with five items seperated with space.
It includes model name, data id(0/1), number of total examples in the experiment, success rate of reduction with label only, and success rate of reduction with probability respectively.

## Parameters for deletion inference:
```
python DeletionInference.py [-h] [--m M] [--NN NN] [--data DATA] [--label LABEL] [--model MODEL] [--lr LR] [--T T] [--TT TT] [--seed SEED]

optional arguments:  
-h, --help     show this help message and exit
  --m M          number of examples in a batch
  --NN NN        total number of examples in the experiment
  --data DATA    which data. 0=CIFAR100, 1=CIFAR10
  --label LABEL  infer with/without label. 0=without label(Del-Inf-Ins), 1=with label(Del-Inf-Exm)
  --model MODEL  target model type
  --lr LR        learning rate
  --T T          number of experimental trails
  --TT TT        number of training epochs
  --seed SEED    random seed
```

- For Fig.1, set '--model cnn'.
- For Fig.2, set '--model vgg'.
- Output: output is written in one line of the file 'deletion.log', with five items seperated with space.
It includes model name, data id(0/1), whether infer with label, number of total examples in the experiment, and the success rate of the deletion inference.
