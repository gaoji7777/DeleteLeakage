from langmodel import *
import numpy as np

dataset = read_sentences_from_file("./ptb.txt")

model = UnigramLanguageModel(dataset, smoothing=True)

a1 = 0
ec = 0
tt = 0
f1_avg = 0

def getmultiset(wseq):
    dct = {}
    for i in wseq:
        if i in dct:
            dct[i] += 1
        else:
            dct[i] = 1
    multiset = []
    for i in dct:
        for j in range(dct[i]):
            multiset.append(i+str(j))
    multiset = set(multiset)
    return multiset
    
def getf1(wt, st):
    wmulti = getmultiset(wt)
    if not '<s>0' in wmulti:
        wmulti.add('<s>0')
    elif not '</s>' in wmulti:
        wmulti.add('</s>0')
    smulti = getmultiset(st)
    tp = len(wmulti.union(smulti))
    fp = len(wmulti.difference(smulti))
    tn = len(smulti.difference(wmulti))
    return tp/(tp + (fp+tn) * 0.5) 
    
T = 1000
for ii in range(T):
    
    i = np.random.randint(len(dataset))
    st = dataset[i]
    del_model = UnigramLanguageModel(dataset[:i] + dataset[i+1:], smoothing=True)

    ws = []
    for word in model.unigram_frequencies.keys():
        if (model.calculate_unigram_probability(word) >= del_model.calculate_unigram_probability(word)):
            ws.append(word)

        
    f1 = getf1(ws, st)
    f1_avg += f1

print(f1_avg/T)
    