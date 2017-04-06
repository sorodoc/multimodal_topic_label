import numpy as np

def calc_trigram(s):
    s = s.replace(' ', '_')
    s_pad = '^' + s + '$'
    trigrams = {}
    for i in range(len(s_pad) - 2):
        tri = s_pad[i:i+3]
        if tri not in trigrams:
            trigrams[tri] = 0
        trigrams[tri] += 1
    return trigrams

def calc_topic():
    fin = open('data/topics.csv', 'r')
    topics = {}
    topic_tri = {}
    for line in fin:
        els = line.strip().split(',')
        topics[els[0]] = els[2:]
    for el in topics:
        topic_tri[el] = {}
        for el1 in topics[el]:
            tri = calc_trigram(el1)
            for el2 in tri:
                if el2 not in topic_tri[el]:
                    topic_tri[el][el2] = 0.0
                topic_tri[el][el2] += tri[el2]
    return topic_tri

def calc_label():
    fin = open('data/crowdflower_dataset.csv', 'r')
    count = 0
    cosine = {}
    topic_tri = calc_topic()

    for line in fin:
        if count == 0:
            count += 1
            continue
        els = line.strip().split('\t')
        l = els[0]
        t_id = int(float(els[1]))
        l_tri = calc_trigram(l)
        t_tri = topic_tri[str(t_id)]
        cosine[(t_id, l)] = calc_cosine(l_tri, t_tri)
    return cosine

def calc_caption(topic, caption):
    l_tri = calc_trigram(caption)
    return calc_cosine(l_tri, topic)

def calc_cosine(x1, x2):
    words = {}
    i = 0
    for el in x1:
        if el not in words:
            words[el] = i
            i += 1
    for el in x2:
        if el not in words:
            words[el] = i
            i += 1
    word_list = list(words.keys())
    a = np.zeros(len(word_list))
    b = np.zeros(len(word_list))
    for el in x1:
        ind = words[el]
        a[ind] += x1[el]
    for el in x2:
        ind = words[el]
        b[ind] += x2[el]
    sim = np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))
    return sim

if __name__ == '__main__':
    calc_label()
