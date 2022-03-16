from langmodel import *
import numpy as np
import time
dataset = read_sentences_from_file("./ptb.txt")

model = BigramLanguageModel(dataset, smoothing=True)

a1 = 0
ec = 0
tt = 0

def dfs(nextp, curr_visited, curr_visited_s, curr, origt, curr_cost, best, th=5):
    if curr == end:
        if curr_cost <= best[1] and (len(curr_visited) - curr_cost >= len(nextp)):
            best[0] = curr_visited[:]
            best[1] = curr_cost
        return
    if time.time() > origt + 10:
        return
    for i in range(len(nextp[curr])):
        if not (nextp[curr][i] in curr_visited_s):
            curr_pre = curr_visited[:]
            curr_visited_s_pre = curr_visited_s.copy()
            curr_visited.append(nextp[curr][i])
            curr_visited_s.add(nextp[curr][i])
            dfs(nextp, curr_visited, curr_visited_s, nextp[curr][i], origt, curr_cost, best, th)
            curr_visited = curr_pre
            curr_visited_s = curr_visited_s_pre
        elif curr_cost<th:
            curr_pre = curr_visited[:]
            curr_visited_s_pre = curr_visited_s.copy()
            curr_visited.append(nextp[curr][i])
            curr_visited_s.add(nextp[curr][i])
            dfs(nextp, curr_visited, curr_visited_s, nextp[curr][i], origt, curr_cost + 1, best, th)
            curr_visited = curr_pre
            curr_visited_s = curr_visited_s_pre
    return
        
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
    
freq_words_top = sorted(model.unigram_frequencies.items(),reverse=True, key = lambda kv: kv[1])[:50]
freq_words = [freq_word_it[0] for freq_word_it in freq_words_top]

f1_avg = 0

T = 1000
for idx in range(T):
    dt_idx = np.random.randint(len(dataset))
    st = dataset[dt_idx]
    del_model = BigramLanguageModel(dataset[:dt_idx] + dataset[dt_idx+1:], smoothing=True)
    
    words = []
    
    for word in model.unigram_frequencies:
       if (model.calculate_unigram_probability(word) > del_model.calculate_unigram_probability(word)):
          words.append(word)
    
    for freq_word in freq_words:
        if not freq_word in set(words):
            words.append(freq_word)
    nd = []
    for pre in words:
        for word in words:
            if(model.calculate_bigram_probabilty(pre, word) > del_model.calculate_bigram_probabilty(pre, word)):
                nd.append((pre, word))
    nextp = []
    start = None
    end = None
    for i in range(len(nd)):
        nextp.append([])
        if nd[i][0] == '<s>':
            start = i
        if nd[i][1] == '</s>':
            end = i
        for j in range(len(nd)):
            if nd[i][1] == nd[j][0]:
                nextp[i].append(j)
        
    if start==None or end==None:
        wt = ['<s>', '</s>']
    else:
        curr_visited = [start]
        curr_visited_s = set([start])
        best = [[], 6]
        origt = time.time()
        dfs(nextp, curr_visited, curr_visited_s,  start, origt, 0, best)
        wt_seq = best[0]
    
        wt = ['<s>'] + [nd[idx][1] for idx in wt_seq]

    if wt == st:
        a1 += 1
        
    f1_avg += getf1(wt, st)   
print(a1/T, f1_avg/T)
