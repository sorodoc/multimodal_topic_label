import tensorflow
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy import array
import numpy as np
import random
from create_negative_visual import create_neg
import shelve


def visual_embs():
    path = '/home/sorodoc/topic_nns/visual_emb/'
    path2 = '/home/sorodoc/topic_nns/textual_embeddings/'
    fin = open(path + 'visual_embeddings_n.txt', 'r')
    v_embs = {}
    c_w_embs = {}
    c_d_embs = {}
    fin2 = open(path2 + 'caption_w_embeddings_n.txt', 'r')
    fin3 = open(path2 + 'caption_d_embeddings_n.txt', 'r')
    for line in fin:
        els = line.split('\t')
        x = array(els[1:])
        x1 = x.astype(np.float)
        v_embs[els[0]] = x1
    for line in fin2:
        els = line.split('\t')
        x = array(els[1:])
        x1 = x.astype(np.float)
        c_w_embs[els[0]] = x1
    for line in fin3:
        els = line.split('\t')
        x = array(els[1:])
        x1 = x.astype(np.float)
        c_d_embs[els[0]] = x1
    return v_embs, c_w_embs, c_d_embs
