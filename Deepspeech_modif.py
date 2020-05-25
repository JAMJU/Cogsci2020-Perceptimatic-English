#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by Juliette MILLET (adapted from Deepspeech 0.4.1)
    Date 12 april 2019
    Script to get the output at different layer level from a deepspeech checkpoint given
    This script needs to be included in the deepspeech folder of the right version
"""
from __future__ import absolute_import, division, print_function

import os
import sys

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'

import numpy as np

import tensorflow as tf

from util.audio import audiofile_to_input_vector
from util.config import Config, initialize_globals

from util.flags import create_flags, FLAGS
from util.logging import log_info, log_error, log_debug, log_warn



# Graph Creation
# ==============

def variable_on_worker_level(name, shape, initializer):
    r'''
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_worker_level()``
    used to create a variable in CPU memory.
    '''
    # Use the /cpu:0 device on worker_device for scoped operations
    if len(FLAGS.ps_hosts) == 0:
        device = Config.worker_device
    else:
        device = tf.train.replica_device_setter(worker_device=Config.worker_device, cluster=Config.cluster)

    with tf.device(device):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def BiRNN(batch_x, seq_length, dropout, reuse=False, batch_size=None, n_steps=-1, previous_state=None, tflite=False, layer_out_wanted = ''):
    r'''
    That done, we will define the learned variables, the weights and biases,
    within the method ``BiRNN()`` which also constructs the neural network.
    The variables named ``hn``, where ``n`` is an integer, hold the learned weight variables.
    The variables named ``bn``, where ``n`` is an integer, hold the learned bias variables.
    In particular, the first variable ``h1`` holds the learned weight matrix that
    converts an input vector of dimension ``n_input + 2*n_input*n_context``
    to a vector of dimension ``n_hidden_1``.
    Similarly, the second variable ``h2`` holds the weight matrix converting
    an input vector of dimension ``n_hidden_1`` to one of dimension ``n_hidden_2``.
    The variables ``h3``, ``h5``, and ``h6`` are similar.
    Likewise, the biases, ``b1``, ``b2``..., hold the biases for the various layers.
    '''
    layers = {}

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    if not batch_size:
        batch_size = tf.shape(batch_x)[0]

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2, 3])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, Config.n_input + 2*Config.n_input*Config.n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)
    layers['input_reshaped'] = batch_x

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    b1 = variable_on_worker_level('b1', [Config.n_hidden_1], tf.zeros_initializer())
    h1 = variable_on_worker_level('h1', [Config.n_input + 2*Config.n_input*Config.n_context, Config.n_hidden_1], tf.contrib.layers.xavier_initializer())
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), FLAGS.relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))
    layers['layer_1'] = layer_1

    # 2nd layer
    b2 = variable_on_worker_level('b2', [Config.n_hidden_2], tf.zeros_initializer())
    h2 = variable_on_worker_level('h2', [Config.n_hidden_1, Config.n_hidden_2], tf.contrib.layers.xavier_initializer())
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), FLAGS.relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))
    layers['layer_2'] = layer_2

    # 3rd layer
    b3 = variable_on_worker_level('b3', [Config.n_hidden_3], tf.zeros_initializer())
    h3 = variable_on_worker_level('h3', [Config.n_hidden_2, Config.n_hidden_3], tf.contrib.layers.xavier_initializer())
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), FLAGS.relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))
    layers['layer_3'] = layer_3

    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell:
    if not tflite:
        fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(Config.n_cell_dim, reuse=reuse)
        layers['fw_cell'] = fw_cell
    else:
        fw_cell = tf.nn.rnn_cell.LSTMCell(Config.n_cell_dim, reuse=reuse)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM RNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [n_steps, batch_size, Config.n_hidden_3])
    if tflite:
        # Generated StridedSlice, not supported by NNAPI
        #n_layer_3 = []
        #for l in range(layer_3.shape[0]):
        #    n_layer_3.append(layer_3[l])
        #layer_3 = n_layer_3

        # Unstack/Unpack is not supported by NNAPI
        layer_3 = tf.unstack(layer_3, n_steps)

    # We parametrize the RNN implementation as the training and inference graph
    # need to do different things here.
    if not tflite:
        output, output_state = fw_cell(inputs=layer_3, dtype=tf.float32, sequence_length=seq_length, initial_state=previous_state)
    else:
        output, output_state = tf.nn.static_rnn(fw_cell, layer_3, previous_state, tf.float32)
        output = tf.concat(output, 0)

    # Reshape output from a tensor of shape [n_steps, batch_size, n_cell_dim]
    # to a tensor of shape [n_steps*batch_size, n_cell_dim]
    output = tf.reshape(output, [-1, Config.n_cell_dim])
    layers['rnn_output'] = output
    layers['rnn_output_state'] = output_state

    # Now we feed `output` to the fifth hidden layer with clipped RELU activation and dropout
    b5 = variable_on_worker_level('b5', [Config.n_hidden_5], tf.zeros_initializer())
    h5 = variable_on_worker_level('h5', [Config.n_cell_dim, Config.n_hidden_5], tf.contrib.layers.xavier_initializer())
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(output, h5), b5)), FLAGS.relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))
    layers['layer_5'] = layer_5

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_worker_level('b6', [Config.n_hidden_6], tf.zeros_initializer())
    h6 = variable_on_worker_level('h6', [Config.n_hidden_5, Config.n_hidden_6], tf.contrib.layers.xavier_initializer())
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)
    layers['layer_6'] = layer_6

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6_resh = tf.reshape(layer_6, [n_steps, batch_size, Config.n_hidden_6], name="raw_logits")
    layers['raw_logits'] = layer_6_resh

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6 if layer_out_wanted == 'layer_6' else layer_5 if layer_out_wanted=='layer_5' else \
        output if layer_out_wanted== 'output_rnn' else layer_6_resh, layers




# Logging
# =======


def create_inference_graph(batch_size=1, n_steps=16, tflite=False, layer_wanted = '', softmax_applied = False):
    # Input tensor will be of shape [batch_size, n_steps, 2*n_context+1, n_input]
    input_tensor = tf.placeholder(tf.float32, [batch_size, n_steps if n_steps > 0 else None, 2*Config.n_context+1, Config.n_input], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

    if not tflite:
        previous_state_c = variable_on_worker_level('previous_state_c', [batch_size, Config.n_cell_dim], initializer=None)
        previous_state_h = variable_on_worker_level('previous_state_h', [batch_size, Config.n_cell_dim], initializer=None)
    else:
        previous_state_c = tf.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_c')
        previous_state_h = tf.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_h')

    previous_state = tf.contrib.rnn.LSTMStateTuple(previous_state_c, previous_state_h)

    no_dropout = [0.0] * 6

    # This is not going to give necessarily the logits, but the output of the layer_wanted
    logits, layers = BiRNN(batch_x=input_tensor,
                           seq_length=seq_length if FLAGS.use_seq_length else None,
                           dropout=no_dropout,
                           batch_size=batch_size,
                           n_steps=n_steps,
                           previous_state=previous_state,
                           tflite=tflite,
                           layer_out_wanted = layer_wanted)

    # TF Lite runtime will check that input dimensions are 1, 2 or 4
    # by default we get 3, the middle one being batch_size which is forced to
    # one on inference graph, so remove that dimension
    if tflite:
        logits = tf.squeeze(logits, [1])

    # Apply softmax for CTC decoder
    if softmax_applied:
        logits = tf.nn.softmax(logits)

    new_state_c, new_state_h = layers['rnn_output_state']

    # Initial zero state
    if not tflite:
        zero_state = tf.zeros([batch_size, Config.n_cell_dim], tf.float32)
        initialize_c = tf.assign(previous_state_c, zero_state)
        initialize_h = tf.assign(previous_state_h, zero_state)
        initialize_state = tf.group(initialize_c, initialize_h, name='initialize_state')
        with tf.control_dependencies([tf.assign(previous_state_c, new_state_c), tf.assign(previous_state_h, new_state_h)]):
            logits = tf.identity(logits, name='logits')

        return (
            {
                'input': input_tensor,
                'input_lengths': seq_length,
            },
            {
                'outputs': logits,
                'initialize_state': initialize_state,
            },
            layers
        )
    else:
        logits = tf.identity(logits, name='logits')
        new_state_c = tf.identity(new_state_c, name='new_state_c')
        new_state_h = tf.identity(new_state_h, name='new_state_h')

        return (
            {
                'input': input_tensor,
                'previous_state_c': previous_state_c,
                'previous_state_h': previous_state_h,
            },
            {
                'outputs': logits,
                'new_state_c': new_state_c,
                'new_state_h': new_state_h,
            },
            layers
        )


def write_fea_file(ar, folder_ZS2015, file_to_save, stride_size_s, win_len_s):
    """
    Function to write outputs as .fea files needed for the Zerospeech challenge evaluation
    :param ar:
    :param folder_ZS2015:
    :param file_to_save:
    :param stride_size_s:
    :param win_len_s:
    :return:
    """
    N = ar.shape[0]
    time_values = np.asarray([[float(i)*stride_size_s + win_len_s/2.] for i in range(N)])
    final_fea = np.concatenate((time_values, ar), axis=1)
    np.savetxt(os.path.join(folder_ZS2015, file_to_save + '.fea'), final_fea, delimiter='\t')

def do_single_file_inference( checkpoint_dir, input_file_path, layer_wanted, softmax_wanted, save_filename, save_folder, stride_size_s, win_size_s, fea_format, csv_format):
    with tf.Session(config=Config.session_config) as session:
        inputs, outputs, _ = create_inference_graph(batch_size=1, n_steps=-1, layer_wanted=layer_wanted, softmax_applied=softmax_wanted)

        # Create a saver using variables from the above newly created graph
        mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
        saver = tf.train.Saver(mapping)

        # Restore variables from training checkpoint
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if not checkpoint:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(checkpoint_dir))
            exit(1)
        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        session.run(outputs['initialize_state'])

        # transformation of the audio file
        features = audiofile_to_input_vector(input_file_path, Config.n_input, Config.n_context)
        #print(features.shape)
        num_strides = len(features) - (Config.n_context * 2)

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future)
        window_size = 2*Config.n_context+1
        features = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, Config.n_input),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False)

        # This is not the logits but the ouput of the layer wanted
        logits = session.run(outputs['outputs'], feed_dict = {
            inputs['input']: [features],
            inputs['input_lengths']: [num_strides],
        })

        logits = np.squeeze(logits)
        if fea_format:
            write_fea_file(logits, save_folder, save_filename, stride_size_s=stride_size_s, win_len_s=win_size_s)
        if csv_format:
            np.savetxt(save_folder + '/' + save_filename + '.csv', logits, delimiter=',')


def total_inference(input_folder, output_folder, checkpoint_dir, layer_wanted, softmax_wanted, win_size_s, stride_size_s, fea_format, csv_format):
    initialize_globals()


    for root, dirs, files in os.walk(input_folder):
        nb = len(files)
        it = 0
        # Iterate over files
        for filename in files:
            print(filename)
            if ((it + 1) % 100) == 0:
                print(it, 'on', nb)
            it += 1
            if not filename.endswith('.wav'):
                continue
            else:
                full_name = os.path.join(root.lstrip('./'), filename)
                tf.reset_default_graph()
                do_single_file_inference(checkpoint_dir=checkpoint_dir, input_file_path='/' + full_name, layer_wanted=layer_wanted,
                                         win_size_s=win_size_s, stride_size_s=stride_size_s,
                                         save_folder=output_folder, save_filename=filename[:-4], softmax_wanted=softmax_wanted,
                                         fea_format=fea_format, csv_format=csv_format)


def main(_):
    import argparse
    parser = argparse.ArgumentParser(description='Inference of the deepspeech model on some data')
    parser.add_argument('input_folder', metavar='inpf', type=str,
                        help='folder where the input data are')
    parser.add_argument('checkpoint_dir', metavar='check', type=str,
                        help='where the model checkpoint is')
    parser.add_argument('layer_wanted', metavar='layer', type=str,
                        help='layer wanted for the output, can be layer_6 or layer_5, or output_rnn, and layer_6_resh by default')
    parser.add_argument('softmax_wanted', metavar= 'softmax', type=str,
                        help='True if softmax applied to output False otherwise')
    parser.add_argument('save_folder', metavar='savf', type=str,
                        help='folder where to save the outputs')
    parser.add_argument('stride_size_s', metavar='stride', type=float,
                        help='stride size of the output') # For deepspeech of this version = 0.02
    parser.add_argument('win_size_s', metavar='window', type=float,
                        help='window size of the output') # For deepspeech of this version = 0.032
    parser.add_argument('fea_format', metavar='fea_format', type=str, help='True if fea format from ZS2015 wanted, False otherwise')
    parser.add_argument('csv_format', metavar='csv_format', type=str,
                        help='True if csv format wanted, False otherwise')
    args = parser.parse_args()

    total_inference(input_folder=args.input_folder, output_folder=args.save_folder, checkpoint_dir=args.checkpoint_dir,
                    layer_wanted=args.layer_wanted, win_size_s=args.win_size_s, stride_size_s=args.stride_size_s,
                    softmax_wanted=True if args.softmax_wanted== 'True' else False ,
                    fea_format= True if args.fea_format == 'True' else False,
                    csv_format=True if args.csv_format== 'True' else False)

if __name__ == '__main__' :
    create_flags()
    tf.app.run(main)