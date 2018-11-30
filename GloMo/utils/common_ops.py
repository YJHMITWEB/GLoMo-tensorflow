# Author: YJHMITWEB
# E-mail: yjhmitweb@gmail.com

import numpy as np
import tensorflow as tf

def Conv2D(inputs, oc, k=3, s=1, p='SAME', d=1, with_bn=False, act='relu', use_bias=False, name='Conv2D'):
    """

    :param inputs: input features
    :param oc: output_channels
    :param k: kernel size
    :param s: strides
    :param p: padding mode, 'VALID' or 'SAME'
    :param d: dilation rate
    :param with_bn: follow the order: conv->bn
    :param act: follow the order: conv->relu or conv->bn->relu if bn is used
    :param name: namescope of conv op
    :return:
    """
    outputs = inputs
    outputs = tf.layers.conv2d(
        inputs=outputs, filters=oc, kernel_size=k, strides=s, dilation_rate=d, padding=p, use_bias=use_bias, name=name)
    if with_bn:
        outputs = tf.layers.batch_normalization(outputs, name=name + '_bn')
    if act:
        if act == 'relu':
            outputs = tf.nn.relu(outputs, name=name + '_relu')
    return outputs

def RNN(inputs, state, cell, cell_state_size, batch_size, num_layers=1, name='RNN'):
    """

    :param inputs: [batch_size, num_steps, cell_state_size]
    :param cell_state_size:
    :param batch_size:
    :param num_layers:
    :return: [batch_size, num_steps, cell_state_size]
    """
    with tf.variable_scope(name):
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs,
                                                        initial_state=state)
        return rnn_outputs

def BN(inputs, axis=-1, name='BN'):
    """

    :param inputs:
    :param axis: Operating Batch Normalization along which axis
    :param name:
    :return:
    """
    return tf.layers.batch_normalization(inputs, axis=axis, name=name)

def Dense(inputs, units, use_bias, act, name='Dense'):
    """

    :param inputs:
    :param units:
    :param use_bias:
    :param act:
    :param name:
    :return:
    """
    if act == 'relu':
        return tf.layers.dense(inputs, units=units, use_bias=use_bias, activation=tf.nn.relu, name=name)
    else:
        return tf.layers.dense(inputs, units=units, use_bias=use_bias, name=name)