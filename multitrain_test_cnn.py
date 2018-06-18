from __future__ import print_function
import os, sys, time, argparse
import h5py, lasagne, theano
import numpy as np
import theano.tensor as T
from collections import defaultdict
from hdf5_deeplearn_utils import MultiHDF5VisualIterator
from lasagne_utils import get_layer_output_fn, store_in_log, save_model, load_log, load_model

def replace_updates_nans_with_zero(updates):
    import theano.tensor as T
    import numpy as np
    # Replace all nans with zeros
    for k,v in updates.items():
        k = T.switch(T.eq(v, np.nan), float(0.), v)
        k = T.switch(T.gt(v**2, 10), v/v * 10., v)
    return updates

def print_model(model):
    print('All parameters:')
    for layer_idx, layer in enumerate(lasagne.layers.get_all_layers(model)):
        print('Layer {: >2}: {}'.format(layer_idx, layer.__class__))
        for param, options in layer.params.items():
            print('\t\t{}: Size: {}'.format(param, param.get_value().shape))

def print_model_with_data(model, data):
    print('All parameters:')
    for layer_idx, layer in enumerate(lasagne.layers.get_all_layers(model)):
        print('Layer {: >2}: {}'.format(layer_idx, layer.__class__))
        for param, options in layer.params.items():
            print('\t\t{}: Size: {}'.format(param, param.get_value().shape))
        print('\t\tOutput data shape: {}'.format(data[layer_idx].shape))

def get_train_and_test_fn(inputs, target_var, network):
    train_prediction = lasagne.layers.get_output(network)
    # Say correct if it is within 45 degrees
    train_correctness = T.abs_(train_prediction-target_var) < 20
    train_loss = lasagne.objectives.squared_error(train_prediction, target_var).mean()
    train_acc = T.mean(train_correctness)

    # Get trainable parameters, and calculate adam updates
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = replace_updates_nans_with_zero(lasagne.updates.adam(train_loss, params, learning_rate=1e-3))
    # Get test prediction
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_correctness = T.abs_(test_prediction-target_var) < 20
    test_loss = lasagne.objectives.squared_error(train_prediction, target_var).mean()
    test_acc = T.mean(test_correctness)

    # Compile the functions
    fn_inputs = inputs + [target_var]
    train_fn = theano.function(fn_inputs, [train_loss, train_acc], updates=updates)
    test_fn = theano.function(fn_inputs, [test_loss, test_acc])

    return train_fn, test_fn

def get_simple_driving_cnn(vid_in, dims=(60, 80), dense_sizes=[64, 64]):
    # INPUTS
    #   (batch size, max sequence length, channels, rows, cols)
    l_vid_in = lasagne.layers.InputLayer(shape=(None, 1)+dims, input_var=vid_in)
    batch_size = vid_in.shape[0]

    # Simple ConvNet
    l_cnn = lasagne.layers.Conv2DLayer(l_vid_in,
                                        num_filters=16, filter_size=(3, 3),
                                        nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l_cnn = lasagne.layers.MaxPool2DLayer(l_cnn, pool_size=(2, 2))
    l_cnn = lasagne.layers.Conv2DLayer(l_cnn,
                                       num_filters=16, filter_size=(3, 3),
                                       nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l_cnn = lasagne.layers.MaxPool2DLayer(l_cnn, pool_size=(2, 2))
    l_cnn = lasagne.layers.Conv2DLayer(l_cnn,
                                       num_filters=16, filter_size=(3, 3),
                                       nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l_cnn = lasagne.layers.MaxPool2DLayer(l_cnn, pool_size=(2, 2))
    l_cnn = lasagne.layers.Conv2DLayer(l_cnn,
                                       num_filters=16, filter_size=(3, 3),
                                       nonlinearity=lasagne.nonlinearities.leaky_rectify)
    # Dense layers
    for dense_size in dense_sizes:
        l_cnn = lasagne.layers.DenseLayer(l_cnn, num_units=dense_size,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_cnn = lasagne.layers.DenseLayer(l_cnn, num_units=1, nonlinearity=None)
    return l_cnn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a driving network.')
    # File and path naming stuff
    parser.add_argument('--h5files',    nargs='+', default='/home/dneil/h5fs/driving/rec1487864316_bin5k.hdf5', help='HDF5 File that has the data.')
    parser.add_argument('--run_id',       default='default', help='ID of the run, used in saving.')
    parser.add_argument('--filename',     default='driving_cnn_19.4_multi', help='Filename to save model and log to.')
    parser.add_argument('--resume',       default=None, help='Filename to load model and log from.')
    # Control meta parameters
    parser.add_argument('--seed',         default=42, type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--batch_size',   default=128, type=int, help='Batch size.')
    parser.add_argument('--num_epochs',   default=100, type=int, help='Number of epochs to train for.')
    parser.add_argument('--patience',     default=4, type=int, help='How long to wait for an increase in validation error before quitting.')
    parser.add_argument('--patience_key', default='test_acc', help='What key to look at before quitting.')
    parser.add_argument('--wait_period',  default=10, type=int, help='How long to wait before looking for early stopping.')
    parser.add_argument('--dataset_keys',  nargs='+', default='aps_frame_48x64', help='Which dataset key (APS, DVS, etc.) to use.')
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Set the save name
    comb_filename = '_'.join([args.filename, args.run_id])

    # Load dataset
    h5fs = [h5py.File(h5file, 'r') for h5file in args.h5files]

    # Create symbolic vars
    vid_in = T.ftensor4('vid_in')
    targets = T.fmatrix('targets')

    # Build model
    print("Building network ...")
    #   Get input dimensions
    network = get_simple_driving_cnn(vid_in)
    # Instantiate log
    log = defaultdict(list)
    print("Built.")

    # Resume if desired
    if args.resume:
        print('RESUMING: {}'.format(args.resume))
        load_model(args.resume, network)
        log = load_log(args.resume)

    # Compile the learning functions
    print('Compiling functions...')
    train_fn, test_fn = get_train_and_test_fn([vid_in], targets, network)
    print('Compiled.')

    # Precalc for announcing
    num_train_batches = int(np.ceil(float(np.sum([len(h5f['train_idxs']) for h5f in h5fs]))/args.batch_size))
    num_test_batches = int(np.ceil(float(np.sum([len(h5f['test_idxs']) for h5f in h5fs]))/args.batch_size))

    # Instantiate iterator
    d = MultiHDF5VisualIterator()

    # Dump some debug data if we like
    # print_model(network)
    out_fn = get_layer_output_fn([vid_in], network)
    temp = MultiHDF5VisualIterator()
    for data in temp.flow(h5fs, args.dataset_keys, 'train_idxs', batch_size=16, shuffle=True):
        vid_in, bY = data
        break
    print(bY.shape)
    print_model_with_data(network, out_fn(vid_in))

    # Finally, launch the training loop.
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print("Starting {} of {}.".format(epoch + 1, args.num_epochs))
        train_err, train_acc, train_batches = 0, 0, 0
        start_time = time.time()

        # Call the data generator
        data_start_time = time.time()
        for data in d.flow(h5fs, args.dataset_keys, 'train_idxs', batch_size=args.batch_size, shuffle=True):
            data_prep_time = time.time() - data_start_time
            vid_in, bY = data
            # Do a training batch
            calc_start_time = time.time()
            err, acc = train_fn(vid_in, bY)
            calc_time = time.time() - calc_start_time
            # Accumulate metadata
            train_err += err
            train_acc += acc
            train_batches += 1
            # Log and print
            log = store_in_log(log, {'b_train_err': err, 'b_train_acc' : acc})
            print("\tBatch {} of {}: ".format(train_batches, num_train_batches, end=""))
            print("Loss: {:.3e} | Acc: {:2.2f}% | Data: {:.3f}s | Calc: {:.3f}s".format(
                float(err), acc*100., data_prep_time, calc_time))
            # Force it to go to output now rather than holding
            sys.stdout.flush()
            data_start_time = time.time()
        print("Training loss:\t\t{:.6f}".format(train_err / train_batches))

        # And a full pass over the test data:
        test_err, test_acc, test_batches = 0, 0, 0
        data_start_time = time.time()

        for data in d.flow(h5fs, args.dataset_keys, 'test_idxs', batch_size=args.batch_size, shuffle=False):
            data_prep_time = time.time() - data_start_time
            vid_in, bY = data
            # Do a test batch
            calc_start_time = time.time()
            err, acc = test_fn(vid_in, bY)
            calc_time = time.time() - calc_start_time
            # Accumulate metadata
            test_err += err
            test_acc += acc
            test_batches += 1
            # Log and print
            log = store_in_log(log, {'b_test_err': err, 'b_test_acc' : acc})
            print("\tBatch {} of {}: ".format(test_batches, num_test_batches, end=""))
            print("Loss: {:.3e} | Acc: {:2.2f}% | Data: {:.3f}s | Calc: {:.3f}s".format(
                float(err), acc*100., data_prep_time, calc_time))
            # Force it to go to output now rather than holding
            sys.stdout.flush()
            data_start_time = time.time()

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.num_epochs, time.time() - start_time))
        # And we store
        log = store_in_log(log, {'test_err': test_err / test_batches,
                             'train_err': train_err / train_batches,
                             'test_acc':test_acc / test_batches*100.,
                             'train_acc':train_acc / train_batches*100.} )

        print("\t Training loss:\t\t{:.6f}".format(log['train_err'][-1]))
        print("\t Test loss:\t\t{:.6f}".format(log['test_err'][-1]))
        print("\t Training accuracy:\t\t{:.2f}".format(log['train_acc'][-1]))
        print("\t Test accuracy:\t\t{:.2f}".format(log['test_acc'][-1]))

        # Save result
        save_model(comb_filename, 'recent', network, log)

        # End if there's no improvement in test error
        best_in_last_set = np.max(log[args.patience_key][-(args.patience-1):])
        # Drop out if our best round was not in the last set, i.e., no improvement
        if len(log[args.patience_key]) > args.wait_period and log[args.patience_key][-args.patience] >= best_in_last_set:
            break
        # Save best-so-far
        if log[args.patience_key][-1] >= np.max(log[args.patience_key]):
            save_model(comb_filename, 'best', network, log)

    # Save result
    save_model(comb_filename, 'final', network, log)
    print('Completed.')
