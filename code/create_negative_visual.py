import random

def group_pop():
    groups = {}
    for i in range(45):
        groups[i] = 'blogs'
    for i in range(45, 83):
        groups[i] = 'iabooks'
    for i in range(83, 143):
        groups[i] = 'news'
    for i in range(143, 228):
        groups[i] = 'pubmed'
    return groups

def create_neg():
    groups = group_pop()
    neg_ex = {'blogs' : [], 'news' : [], 'iabooks' : [], 'pubmed' : []}
    fin = open('data/dataset_images.txt', 'r')
    for line in fin:
        els = line.split('\t')
        t = int(float(els[0]))
        text = els[1]
        if groups[t] == 'blogs':
            neg_ex['pubmed'].append(text)
        if groups[t] == 'iabooks':
            neg_ex['news'].append(text)
        if groups[t] == 'news':
            neg_ex['iabooks'].append(text)
        if groups[t] == 'pubmed':
            neg_ex['blogs'].append(text)
    for el in neg_ex:
        random.shuffle(neg_ex[el])
    return groups, neg_ex
