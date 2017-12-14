#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/11/17 11:50 AM
# @Author  : Andy
# @Site    : 
# @File    : tf_trace.py
# @Software: PyCharm Community Edition

import os
import sys
# https://stackoverflow.com/questions/37751739/tensorflow-code-optimization-strategy/37774430

import tensorflow as tf
from tensorflow.python.client import timeline

x = tf.random_normal([1000, 1000])
y = tf.random_normal([1000, 1000])
res = tf.matmul(x, y)

# Run the graph with full trace option
with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(res, options=run_options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
    
    