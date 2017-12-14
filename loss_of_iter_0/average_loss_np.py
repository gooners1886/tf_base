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

def my_softmax(logits):
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]

    probs = np.zeros([batch_size, num_classes])

    for id_batch in xrange(batch_size):
        logits_exp = np.exp(logits[id_batch])
        sum = np.sum(logits_exp)
        prob = logits_exp / sum

        probs[id_batch] = prob

    return probs


def gen_label(batch_size, num_classes):
    labels_sparse = np.zeros([batch_size, num_classes])
    for id_batch in xrange(batch_size):
        label_hot = np.random.randint(low=0, high=num_classes)
        labels_sparse[id_batch, label_hot] = 1.0

    return labels_sparse


def cal_single_loss(prob, label_sparse):
    sum = np.sum(label_sparse * np.log(prob))

    return sum


def my_loss(probs, labels_sparse, batch_size):
    loss_avg = 0.0
    for id_batch in xrange(batch_size):
        prob_this = probs[id_batch]
        label_this = labels_sparse[id_batch]
        loss_this = (-1.0) * cal_single_loss(prob_this, label_this)

        print ("prob_this = {}\n label_this = {}".format(prob_this, label_this))
        print ("loss_this = {}".format(loss_this))

        loss_avg += loss_this

    loss_avg /= batch_size

    print ("\nloss_avg = {}".format(loss_avg))


def average_loss(batch_size, num_classes, bRandom):
    if bRandom:
        logits = np.random.normal(0.0, 10, size=(batch_size, num_classes))
    else:
        logits = 1.0 / num_classes * np.ones((batch_size, num_classes))

    probs = my_softmax(logits)

    labels_sparse = gen_label(batch_size, num_classes)

    loss = my_loss(probs, labels_sparse, batch_size)

    print ("ln({}) = {}".format(num_classes, np.log(float(num_classes))))


def main(argv):
    batch_size = 4
    num_classes = 40

    bRandom = True

    average_loss(batch_size, num_classes, bRandom)


if __name__ == '__main__':
    main(sys.argv)

