#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import tensorflow as tf

from nets import inception_v4
from nets import inception_utils

slim = tf.contrib.slim


number_of_classes = 764


# Define the model that we want to use -- specify to use only two classes at the last layer
def cam_inception(inputs, num_classes=number_of_classes, is_training=True, reuse=None, delta=0.6):

    with tf.variable_scope('InceptionV4',[inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = inception_v4.inception_v4_base(inputs, scope=scope)

    inception_c_feature = net
    with tf.variable_scope('cam_classifier/A'):
        net = slim.conv2d(inception_c_feature, 1024, [3, 3],
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          padding='SAME',
                          scope='conv1_3x3')
        net = slim.conv2d(net, 1024, [3, 3],
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          padding='SAME',
                          scope='conv2_3x3')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          scope='conv3_1x1')
        end_points['features_A'] = net
        # GAP
        kernel_size = net.get_shape()[1:3]
        if kernel_size.is_fully_defined():
            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a')
        else:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True,
                                 name='global_pool')

        logits = slim.flatten(net, scope='Flatten')
        end_points['Logits'] = logits
        end_points['Predictions_A'] = tf.argmax(logits, 1, name='Predictions_A')

    return logits, end_points

cam_inception.default_image_size = 299
cam_inception_arg_scope = inception_utils.inception_arg_scope

