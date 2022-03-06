import os
import lasagne
import _pickle as pickle
import numpy as np
import theano.tensor as T
# Create 5D tensor type
ftensor5 = T.TensorType(dtype="float32", broadcastable=(False,) * 5)

def save_model(filename, suffix, model, log=None, announce=True, log_only=False):
    # Build filename
    filename = '{}_{}'.format(filename, suffix)
    # Store in separate directory
    filename = os.path.join('./models/', filename)
    # Inform user
    if announce:
        print('Saving to: {}'.format(filename))
    # Generate parameter filename and dump
    param_filename = '%s.params' % (filename)
    if not log_only:
        # Acquire Data
        data = lasagne.layers.get_all_param_values(model)
        with open(param_filename, 'wb') as f:
            pickle.dump(data, f)
    # Generate log filename and dump
    if log is not None:
        log_filename = '%s.log' % (filename)
        with open(log_filename, 'wb') as f:
            pickle.dump(log, f)

def load_model(filename, model):
    # Build filename
    filename = os.path.join('./models/', '%s.params' % (filename))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)
    return model

def load_log(filename, append_dir=True):
    if append_dir:
        filename = os.path.join('./models/', '%s.log' % (filename))
    with open(filename, 'r') as f:
        log = pickle.load(f)
    return log

def store_in_log(log, kv_pairs):
    # Quick helper function to append values to keys in a log
    for k,v in kv_pairs.items():
        log[k].append(v)
    return log

def non_flattening_dense(l_in, batch_size, seq_len, *args, **kwargs):
    # Flatten down the dimensions for everything but the features
    l_flat = lasagne.layers.ReshapeLayer(l_in, (-1, [2]))
    # Make a dense layer connected to it
    l_dense = lasagne.layers.DenseLayer(l_flat, *args, **kwargs)
    # Reshape it back out - this could be done implicitly, but I want to throw an error if not matching
    l_reshaped = lasagne.layers.ReshapeLayer(l_dense, (batch_size, seq_len, l_dense.output_shape[1]))
    return l_reshaped

def get_layer_output_fn(fn_inputs, network, on_unused_input='raise'):
    import theano
    outs = []
    for layer in lasagne.layers.get_all_layers(network):
        outs.append(lasagne.layers.get_output(layer, deterministic=True))
    out_fn = theano.function(fn_inputs, outs, on_unused_input=on_unused_input)
    return out_fn

def get_output_fn(fn_inputs, network):
    import theano
    output = lasagne.layers.get_output(network, deterministic=True)
    out_fn = theano.function(fn_inputs, output)
    return out_fn
