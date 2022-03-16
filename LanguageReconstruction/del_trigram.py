from langmodel import *
import numpy as np
import time

dataset = read_sentences_from_file("./ptb.txt")
model = TrigramLanguageModel(dataset, smoothing=True)

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
    
a1 = 0
f1_avg = 0

freq_words_top = sorted(model.unigram_frequencies.items(),reverse=True, key = lambda kv: kv[1])[:50]
freq_words = [freq_word_it[0] for freq_word_it in freq_words_top]

T = 1000
for idx in range(T):
    ii = np.random.randint(len(dataset))
    st = dataset[ii]
    del_model = TrigramLanguageModel(dataset[:ii] + dataset[ii+1:], smoothing=True)
    
    words = []

    for word in model.unigram_frequencies:
       if (model.calculate_unigram_probability(word) > del_model.calculate_unigram_probability(word)):
          words.append(word)
    
    for freq_word in freq_words:
        if not freq_word in set(words):
            words.append(freq_word)

    nd = []
    for previous_word2 in words:
        for previous_word1 in words:
            for word in words:
                if(model.calculate_trigram_probabilty(previous_word2, previous_word1, word) > del_model.calculate_trigram_probabilty(previous_word2, previous_word1, word)):
                    nd.append((previous_word2, previous_word1, word))

    nextp = []
    start = None
    end = None
    for i in range(len(nd)):
        nextp.append([])
        if nd[i][0] == '<s>':
            start = i
        if nd[i][2] == '</s>':
            end = i
        for j in range(len(nd)):
            if (nd[i][1] == nd[j][0]) and (nd[i][2] == nd[j][1]):
                nextp[i].append(j)

    if start==None or end==None:
        wt = ['<s>', '</s>']
    else:
        curr_visited = [start]
        curr_visited_s = set([start])
        best = [[], 6]
        origt = time.time()
        dfs(nextp, curr_visited, curr_visited_s, start, origt, 0, best)
        wt_seq = best[0]
    
        wt = ['<s>', nd[start][1]] + [nd[idx][2] for idx in wt_seq]
    f1 = getf1(wt, st)
    f1_avg += f1    
    if wt == st:
        a1 += 1

print(a1/T, f1_avg/T)
