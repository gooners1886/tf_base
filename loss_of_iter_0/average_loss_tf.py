#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/11/17 6:25 PM
# @Author  : Andy
# @Site    :
# @File    : loss_average.py
# @Software: PyCharm Community Edition

import os
import sys
import math
import numpy as np


# 证明当模型随机初始化的时候，计算的平均loss比 ln(num_class)要大   即使是在统计意义上，也不一定是在ln(num_class)

import tensorflow as tf

BATCH_SIZE = 4
NUM_CLASSES = 40




def get_logits(batch_size, num_classes, bRandom, i_seed=None):

    if bRandom:
        if i_seed:
            seed = np.random.RandomState(i_seed)
            logits = seed.normal(0.0, 10, size=(batch_size, num_classes))
        else:
            logits = np.random.normal(0.0, 10, size=(batch_size, num_classes))
    else:
        logits = 1.0 / num_classes * np.ones((batch_size, num_classes))

    return logits




def get_labels(batch_size, num_classes, i_seed=None):
    labels_sparse = np.zeros([batch_size, num_classes])
    for id_batch in xrange(batch_size):
        if i_seed:
            seed = np.random.RandomState(i_seed)
            label_hot = seed.randint(low=0, high=num_classes)
        else:
            label_hot = np.random.randint(low=0, high=num_classes)
        labels_sparse[id_batch, label_hot] = 1.0

    return labels_sparse




logits = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
labels_sparse = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
probs = tf.nn.softmax(logits)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_sparse, logits=logits))




init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
logits_this = get_logits(BATCH_SIZE, NUM_CLASSES, True, 0)
labels_this = get_labels(BATCH_SIZE, NUM_CLASSES, 0)
feed_dict = {logits:logits_this, labels_sparse: labels_this}
loss_ = sess.run(loss, feed_dict=feed_dict)

print("loss_ = {}".format(loss_))
