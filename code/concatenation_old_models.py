import tensorflow
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy import array
import numpy as np
import random
from create_negative_visual import create_neg as neg_visual
from create_negative_text import create_neg as neg_textual
from textual_preprocess import read_embeddings as text_emb
from visual_preprocess import visual_embs as visual_emb
import model_concatenation
import shelve
from calculate_trigram import calc_label as calc_trigram
from calculate_trigram import calc_topic
from calculate_trigram import calc_caption

def idcg_at_k(v, k):
    v2 = sorted(v, reverse = True)
    small_v = np.asfarray(v2)[:k]
    return small_v[0] + np.sum(small_v[1:] / np.log2(np.arange(2, small_v.size + 1)))

def dcg_at_k(v, k):
    small_v = np.asfarray(v)[:k]
    return small_v[0] + np.sum(small_v[1:] / np.log2(np.arange(2, small_v.size + 1)))

def ndcg_at_k(v, k):
    return dcg_at_k(v, k) / idcg_at_k(v, k)


def evaluate_fold(v_probs, w_probs, w_t_inf, w_t_l, v_t_inf, v_t_l):
    pred_test = {}
    diff = 0.0
    for i in range(len(v_probs)):
        diff += (3.0 * abs(v_probs[i][0] - v_t_l[i][0]))
        if v_t_inf[i][0] not in pred_test:
            pred_test[v_t_inf[i][0]] = []
        pred_test[v_t_inf[i][0]].append((3.0 * v_probs[i][0], 3.0 * v_t_l[i][0]))
    for i in range(len(w_probs)):
        diff += (3.0 * abs(w_probs[i][0] - w_t_l[i][0]))
        if w_t_inf[i][0] not in pred_test:
            pred_test[v_t_inf[i][0]] = []
        pred_test[w_t_inf[i][0]].append((3.0 * w_probs[i][0], 3.0 * w_t_l[i][0]))
    return pred_test, diff

def evaluate_fold_text(w_probs, w_t_inf, w_t_l):
    pred_test = {}
    diff = 0.0
    for i in range(len(w_probs)):
        diff += (3.0 * abs(w_probs[i][0] - w_t_l[i][0]))
        if w_t_inf[i][0] not in pred_test:
            pred_test[w_t_inf[i][0]] = []
        pred_test[w_t_inf[i][0]].append((3.0 * w_probs[i][0], 3.0 * w_t_l[i][0]))
    return pred_test, diff

def evaluate_fold_visual(v_probs, v_t_inf, v_t_l):
    pred_test = {}
    diff = 0.0
    for i in range(len(v_probs)):
        diff += (3.0 * abs(v_probs[i][0] - v_t_l[i][0]))
        if v_t_inf[i][0] not in pred_test:
            pred_test[v_t_inf[i][0]] = []
        pred_test[v_t_inf[i][0]].append((3.0 * v_probs[i][0], 3.0 * v_t_l[i][0]))
    return pred_test, diff

def calculate_ndcg(pred_test, groups):
    avg_ndcg = 0.0
    avg_top1 = 0.0
    for el in pred_test:
        preds = sorted(pred_test[el], key = lambda x :x[0], reverse = True)
        t_preds = []
        avg_top1 += preds[0][1]
        gr = w_t_groups[int(el)]
        groups[gr]['total'] += 1.0
        groups[gr]['correct'] += preds[0][1]
        for el in preds:
            t_preds.append(el[1])
        d = ndcg_at_k(t_preds, dcg_k)
        avg_ndcg += d
    return avg_top1 / len(pred_test), avg_ndcg / len(pred_test)

def add_w_embs(w_inf, t_w_embs, t_d_embs, l_w_embs, l_d_embs):
    emb_l = 1200
    w_inp = np.zeros((len(w_inf), emb_l))
    for i in range(len(w_inf)):
        emb = np.hstack((t_w_embs[str(w_inf[i][0])], t_d_embs[str(w_inf[i][0])], l_w_embs[w_inf[i][1]], l_d_embs[w_inf[i][1]]))
        for j in range(len(emb)):
            w_inp[i][j]  = emb[j]
    return w_inp

def add_v_embs(v_inf, t_w_embs, t_d_embs, v_embs, c_w_embs, c_d_embs):
    emb_l = 1600
    v_inp = np.zeros((len(v_inf), emb_l))
    c_inp = np.zeros((len(v_inf), 1))
    for i in range(len(v_inf)):
#        emb = np.hstack((t_w_embs[str(v_inf[i][0])], t_d_embs[str(v_inf[i][0])], c_w_embs[v_inf[i][1]], c_d_embs[v_inf[i][1]], v_embs[v_inf[i][1]]))
        emb = np.hstack((t_w_embs[str(v_inf[i][0])], t_d_embs[str(v_inf[i][0])], v_embs[v_inf[i][1]]))
        for j in range(len(emb)):
            v_inp[i][j]  = emb[j]
#        print captions[v_inf[i][1]][0]
        c_inp[i][0] = calc_caption(topic_trigram[str(v_inf[i][0])], captions[v_inf[i][1]][0])
    return v_inp, c_inp

def read_folds():
    k_folds = {}
    fin = open('data/folds.txt', 'r')
    i =  0
    for line in fin:
        els = line.strip().split('\t')
        tr = [int(x) for x in els[0].split()]
        test = [int(x) for x in els[1].split()]
        k_folds[i] = {}
        k_folds[i]['train'] = tr
        k_folds[i]['test'] = test
        i += 1
    return k_folds

def prepare_textual_baseline(fold_nr = 5):
    t_baseline = {}
    fin = open('data/predcombined5times', 'r')
    fin2 = open('data/label_list5times', 'r')
    for i in range(fold_nr):
        t_baseline[i] = {}
        for j in range(4332):
            l = fin2.readline().strip()
            v = fin.readline().strip()
            els = l.split()
            if int(els[1]) not in t_baseline[i]:
                t_baseline[i][int(els[1])] = {}
            t_baseline[i][int(els[1])][els[0]] = float(v)
    return t_baseline

page_scores = {}
trigram_scores = calc_trigram()
topic_trigram = calc_topic()
captions = {}

def read_captions():
    fin = open('data/captions.txt', 'r')
    for line in fin:
        els = line.strip().split('\t')
        captions[els[2]] =  els[3:]

def read_page_rank():
    fin = open('data/page_rank_scores.txt', 'r')
    for line in fin:
        els = line.strip().split('\t')
        page_scores[els[0]] = float(els[1])

def calc_page_rank(w_inf):
    page = np.zeros((len(w_inf), 1))
    for i in range(len(w_inf)):
        page[i][0] = page_scores[w_inf[i][1]]
    return page

def calc_page_trigram(w_inf):
    page = np.zeros((len(w_inf), 2))
    for i in range(len(w_inf)):
        page[i][0] = page_scores[w_inf[i][1]]
        page[i][1] = trigram_scores[(w_inf[i][0], w_inf[i][1])]
    return page

def calculate_prob(info, base):
    probs = np.zeros((len(info), 1))
    i = 0
    for el in info:
        t = el[0]
        l = el[1].replace(" ", "_")
        value = base[t][l] / 3.0
        probs[i][0] = float(value)
        i += 1
#    print probs
    return probs

t_w_embs, t_d_embs, l_w_embs, l_d_embs = text_emb()
v_embs, c_w_embs, c_d_embs = visual_emb()

act_f = "relu"
d_value = 0.2
batch_size = 16
dcg_k = 5
nb_epoch = 10
read_page_rank()
read_captions()

fout = open('results.txt', 'w')
fout.write('activation function : ' + act_f + ' batch size : ' + str(batch_size) + ' drop value : ' + str(d_value) + ' number of epochs : ' + str(nb_epoch) + '\n')
avg_fold_ndcg = 0.0
avg_fold_pr = 0.0
fold_count = 0
avg_fold_top1 = 0.0
avg_fold_ndcg_v = 0.0
avg_fold_ndcg_w = 0.0
avg_fold_top1_v = 0.0
avg_fold_top1_w = 0.0
avg_fold_pr_v = 0.0
avg_fold_pr_w = 0.0
batches = 250

text_base = prepare_textual_baseline()

groups = {'blogs' : {'total' : 0.0, 'correct' : 0.0},
        'news' : {'total' : 0.0, 'correct' : 0.0},
        'iabooks' : {'total' : 0.0, 'correct' : 0.0},
        'pubmed' : {'total' : 0.0, 'correct' : 0.0}}

groups_v = {'blogs' : {'total' : 0.0, 'correct' : 0.0},
        'news' : {'total' : 0.0, 'correct' : 0.0},
        'iabooks' : {'total' : 0.0, 'correct' : 0.0},
        'pubmed' : {'total' : 0.0, 'correct' : 0.0}}

groups_w = {'blogs' : {'total' : 0.0, 'correct' : 0.0},
        'news' : {'total' : 0.0, 'correct' : 0.0},
        'iabooks' : {'total' : 0.0, 'correct' : 0.0},
        'pubmed' : {'total' : 0.0, 'correct' : 0.0}}

#w_shelf = shelve.open('/home/sorodoc/topic_nns/datasets/linguistic_dataset_pos_n.shlf')
#v_shelf = shelve.open('/home/sorodoc/topic_nns/datasets/visual_dataset_n.shlf')

w_shelf = shelve.open('data/text_dataset.shlf')
v_shelf = shelve.open('data/visual_dataset_neg2.shlf')

folds = read_folds()

for f in folds:
    print len(folds)
    fold_count += 1
    d = folds[f]
    print 'fold : ' + str(fold_count)
    v_t_groups, v_neg_ex = neg_visual()
    w_t_groups, w_neg_ex = neg_textual()
#    w_fold = w_shelf['fold_' + str(fold_count)]
#    v_fold = v_shelf['fold_' + str(fold_count)]
    w_fold = w_shelf['fold_' + str(f)]
    v_fold = v_shelf['fold_' + str(f)]
    w_tr_l = w_fold['train_l']
    w_tr_inf = w_fold['train_inf']
    w_tr_inp = add_w_embs(w_tr_inf, t_w_embs, t_d_embs, l_w_embs, l_d_embs)
    w_t_l = w_fold['test_l']
    w_t_inf = w_fold['test_inf']
#    p_tr_inp = calc_page_rank(w_tr_inf)
    p_tr_inp = calc_page_trigram(w_tr_inf)
    w_t_inp = add_w_embs(w_t_inf, t_w_embs, t_d_embs, l_w_embs, l_d_embs)
#    p_t_inp = calc_page_rank(w_t_inf)
    p_t_inp = calc_page_trigram(w_t_inf)
    v_tr_l = v_fold['train_l']
    v_tr_inf = v_fold['train_inf']
    v_tr_inp, caption_tr_inp = add_v_embs(v_tr_inf, t_w_embs, t_d_embs, v_embs, c_w_embs, c_d_embs)
    v_t_l = v_fold['test_l']
    v_t_inf = v_fold['test_inf']
    v_t_inp, caption_t_inp = add_v_embs(v_t_inf, t_w_embs, t_d_embs, v_embs, c_w_embs, c_d_embs)
    m1 = model_concatenation.TNN()
    model1 = m1.buildModel()
#    print w_t_inf
#    print w_t_l
    probs2 = calculate_prob(w_t_inf, text_base[(fold_count - 1) / 10])
    b_size1 = len(v_tr_l) / batches
    b_size2 = len(w_tr_l) / batches
    for j in range(nb_epoch):
        print 'epoch : ' + str(j)
        count_loss1 = 0.0
        loss1 = 0.0
        count_loss2 = 0.0
        loss2 = 0.0
        for i in range(batches):
            st1 = i * b_size1
            fin1 = (i + 1) * b_size1
            loss1 += model1.train_on_batch([v_tr_inp[st1: fin1]], [v_tr_l[st1: fin1]])
            count_loss1 += 1.0
        if len(v_tr_l) % b_size1 > 0:
            st1 = (len(v_tr_l) % b_size1) * -1
            loss1 += model1.train_on_batch([v_tr_inp[st1:]], [v_tr_l[st1:]])
            count_loss1 += 1.0
        print 'visual loss : ' + str(loss1 / count_loss1) + ' batches : ' + str(count_loss1) + ' batch_size : ' + str(b_size1)
#    model1.fit([v_tr_inp], [v_tr_l], nb_epoch = nb_epoch, batch_size = batch_size, validation_split = 0.1)
    probs1 = model1.predict([v_t_inp], batch_size = batch_size)
    pred_test, diff = evaluate_fold(probs1, probs2, w_t_inf, w_t_l, v_t_inf, v_t_l)
    pred_text, diff_text = evaluate_fold_text(probs2, w_t_inf, w_t_l)
    pred_visual, diff_visual = evaluate_fold_visual(probs1, v_t_inf, v_t_l)
    avg_top1_v, avg_ndcg_v = calculate_ndcg(pred_visual, groups_v)
    avg_top1_w, avg_ndcg_w = calculate_ndcg(pred_text, groups_w)
    avg_top1, avg_ndcg = calculate_ndcg(pred_test, groups)

    probs_l = len(probs1) + len(probs2)
    print diff / probs_l
    print 'average ndcg : ' + str(avg_ndcg)
    print avg_top1

    fout.write('average ndcg for fold :' + str(fold_count) + ' is : ' + str(avg_ndcg) +
                   ' average precision : ' + str(diff / probs_l) +
                   ' and average top 1 : ' + str(avg_top1) + '\n')
    fout.write('average text ndcg for fold :' + str(fold_count) + ' is : ' + str(avg_ndcg_w) +
                   ' average precision : ' + str(diff_text / len(probs2)) +
                   ' and average top 1 : ' + str(avg_top1_w) + '\n')
    fout.write('average visual ndcg for fold :' + str(fold_count) + ' is : ' + str(avg_ndcg_v) +
                   ' average precision : ' + str(diff_visual / len(probs1)) +
                   ' and average top 1 : ' + str(avg_top1_v) + '\n')
    fout.write('\n')
    avg_fold_ndcg += avg_ndcg
    avg_fold_pr += (diff / probs_l)
    avg_fold_top1 += avg_top1
    avg_fold_ndcg_v += avg_ndcg_v
    avg_fold_pr_v += (diff_visual / len(probs1))
    avg_fold_top1_v += avg_top1_v
    avg_fold_ndcg_w += avg_ndcg_w
    avg_fold_pr_w += (diff_text / len(probs2))
    avg_fold_top1_w += avg_top1_w

c_f = fold_count
fout.write('\n')
fout.write('average ndcg : ' + str(avg_fold_ndcg / c_f) +
               ' average accuracy : ' + str(avg_fold_pr / c_f) +
                ' average top1 : ' + str(avg_fold_top1 / c_f) + '\n')
fout.write('top1 blogs : ' + str(groups['blogs']['correct'] / groups['blogs']['total']) + '\n')
fout.write('top1 news : ' + str(groups['news']['correct'] / groups['news']['total']) + '\n')
fout.write('top1 iabooks : ' + str(groups['iabooks']['correct'] / groups['iabooks']['total']) + '\n')
fout.write('top1 pubmed : ' + str(groups['pubmed']['correct'] / groups['pubmed']['total']) + '\n')

fout.write('\n')
fout.write('average ndcg visual: ' + str(avg_fold_ndcg_v / c_f) +
               ' average accuracy : ' + str(avg_fold_pr_v / c_f) +
                ' average top1 : ' + str(avg_fold_top1_v / c_f) + '\n')
fout.write('top1 blogs : ' + str(groups_v['blogs']['correct'] / groups_v['blogs']['total']) + '\n')
fout.write('top1 news : ' + str(groups_v['news']['correct'] / groups_v['news']['total']) + '\n')
fout.write('top1 iabooks : ' + str(groups_v['iabooks']['correct'] / groups_v['iabooks']['total']) + '\n')
fout.write('top1 pubmed : ' + str(groups_v['pubmed']['correct'] / groups_v['pubmed']['total']) + '\n')

fout.write('\n')
fout.write('average ndcg textual: ' + str(avg_fold_ndcg_w / c_f) +
               ' average accuracy : ' + str(avg_fold_pr_w / c_f) +
                ' average top1 : ' + str(avg_fold_top1_w / c_f) + '\n')
fout.write('top1 blogs : ' + str(groups_w['blogs']['correct'] / groups_w['blogs']['total']) + '\n')
fout.write('top1 news : ' + str(groups_w['news']['correct'] / groups_w['news']['total']) + '\n')
fout.write('top1 iabooks : ' + str(groups_w['iabooks']['correct'] / groups_w['iabooks']['total']) + '\n')
fout.write('top1 pubmed : ' + str(groups_w['pubmed']['correct'] / groups_w['pubmed']['total']) + '\n')
