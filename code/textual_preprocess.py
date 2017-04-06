import tensorflow
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy import array
import numpy as np
import random
from create_negative_text import create_neg
import shelve
'''
t_w_embs = {}
t_d_embs = {}
label_w_embs = {}
label_d_embs = {}
emb_l = 1200
'''

def read_embeddings():
    t_w_embs = {}
    t_d_embs = {}
    label_w_embs = {}
    label_d_embs = {}
    emb_l = 1200
    path = '/home/sorodoc/topic_nns/textual_embeddings/'
    fin = open(path + 'label_d_embs_n.txt', 'r')
    for line in fin:
        els = line.split('\t')
        x = array(els[1:])
        label_d_embs[els[0]] = x.astype(np.float)
#    print label_d_embs
    fin2 = open(path + 'label_w_embs_n.txt', 'r')
    for line in fin2:
        els = line.split('\t')
        x = array(els[1:])
        label_w_embs[els[0]] = x.astype(np.float)
    fin3 = open(path + 'topic_d_embeddings_n.txt', 'r')
    for line in fin3:
        els = line.split('\t')
        x = array(els[2:])
        t_d_embs[els[0]] = x.astype(np.float)
    fin4 = open(path + 'topic_w_embeddings_n.txt', 'r')
    for line in fin4:
        els = line.split('\t')
        x = array(els[2:])
        t_w_embs[els[0]] = x.astype(np.float)
    return t_w_embs, t_d_embs, label_w_embs, label_d_embs
