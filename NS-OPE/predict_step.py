import numpy as np
import theano
import theano.tensor as T
import os
import sys

from lasagne.layers import *
import lasagne

__author__ = 'idurugkar'

num_units = 1
num_inputs = 1
num_outputs = 1

input_var = T.matrix(name='inputs', dtype=theano.config.floatX)
target_var = T.matrix(name='targets', dtype=theano.config.floatX)


def build_net(inputs):
    l_inp = InputLayer((None, sequence_size), input_var=inputs)  # ((None, None, 1))
    # retrieve symbolic references to batch_size and sequence_length
    batch_size, sequence_length = l_inp.input_var.shape
    l_rshp1 = ReshapeLayer(l_inp, (batch_size, sequence_length, 1))

    l_recurr = GRULayer(l_rshp1, num_units=num_units, learn_init=True)
    # Flatten output of batch and sequence so that each time step
    # of each sequence is processed independently.
    # Didn't understand this part :/
    l_rshp2 = ReshapeLayer(l_recurr, (-1, num_units))
    l_dense = DenseLayer(l_rshp2, num_units=num_outputs, nonlinearity=None)
    l_out = ReshapeLayer(l_dense, (batch_size, sequence_length))

    return l_out

resultsFile = 'Data1/results_basic_50_99.txt'
# with open(resultsFile, 'w') as rf:
#     rf.writelines(['File\tPrediction\n'])
overallFile = 'Data1/overall_basic_50_99.txt'
# with open(overallFile, 'w') as ro:
#     ro.writelines(['File\tMSE\tAbs\n'])

ind = []
evals = []

with open('trueEval.txt', 'r') as fr:
    fr.readline()
    for line in fr.readlines():
        parts = line.split()
        ind.append(int(parts[0]))
        evals.append(float(parts[-1]))
    # evals = [float(line.split()[-1]) for line in fr.readlines()]
# sequence_size = 2000
MSE = 0.
count = 0
# dataset = []

for root, folder, files in os.walk('Data1/IWRs'):
    for f_i in range(50,100):
        # for f in files:
        #     if f.startswith('.'):
        #         continue
        with open('Data1/IWRs/IWRs_out%d.txt' % f_i, 'r') as fp:
            text = fp.read()
        dataset = [float(tok) for tok in text.split()]
        # break

        dataset = np.asarray(dataset, dtype=theano.config.floatX)
        normalizer = dataset.min() * -1
        dataset /= normalizer
        dataset = dataset.reshape((1, -1))
        full_train = np.copy(dataset[:, :-1])
        full_targets = np.copy(dataset[:, 1:])
        # sequence_size = 2000

        # p = np.random.permutation(dataset.shape[0])[:20]
        # dataset = dataset.reshape((-1, sequence_size), order='C')

        num_examples, sequence_size = dataset.shape

        network = build_net(input_var)
        save_network = build_net(input_var)

        # temp = train[1:2]

        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.squared_error(prediction, target_var)
        # reg = lasagne.objectives.squared_error(prediction, input_var)
        # reg = reg.mean() * 0.03
        loss = loss.mean()  # + reg

        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adagrad(loss, params, learning_rate=0.1)
        # updates = lasagne.updates.adadelta(loss, params, learning_rate=0.1, rho=0.99)

        # generate_output = get_output(network, deterministic=True)
        test_prediction = get_output(network, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
        test_loss = test_loss.mean()

        index = T.scalar(name='index', dtype='int64')
        # train_fn = theano.function(inputs=[index], outputs=loss,
        #                            givens={input_var: train[index],
        #                                    target_var: train_targets[index]},
        #                            updates=updates)
        train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
        val_fn = theano.function([input_var, target_var], test_loss, allow_input_downcast=True)

        # selection_range = 2000
        # gap = 10 #train.shape[1]/selection_range
        # print train.shape

        # p = [i*gap for i in range(selection_range)]
        # print p
        min_err = float('inf')
        min_epoch = 0

        num_epochs = 3000
        for epoch in range(num_epochs):
            err = train_fn(full_train, full_targets)
            if err < min_err:
                set_all_param_values(save_network, get_all_param_values(network))
                min_err = err
                min_epoch = epoch
            print('Epoch %d, Error %f' % (epoch, err))
            sys.stdout.flush()

        # pred_err = val_fn(test, test_targets)
        # print('Test error: %f' % pred_err)

        print('Generating output based on epoch %d at error: %f' % (min_epoch, min_err))
        out = get_output(save_network)
        get_it = theano.function([input_var], out, allow_input_downcast=True)

        got_it = get_it(dataset)
        # plt.plot(full_train[0])
        # plt.plot(got_it[0], linestyle='--', color='r')
        # plt.show()
        count += 1

        print('File %d prediction = %f' % (f_i, (got_it[0, -1]*normalizer)))
        print('Target: %f' % evals[-1])
        with open(resultsFile, 'a+') as rf:
            rf.writelines(['%d\t%f\n' % (f_i, (got_it[0, -1]*normalizer))])

        final_error = 0.
        abs_error = 0.
        for i in range(1, len(ind)):
            pred = (got_it[0, ind[i]-1] * normalizer)
            error = (pred - evals[i])
            final_error += error ** 2
            abs_error += abs(error)
        mean_final_error = final_error / len(evals)
        mean_abs = abs_error / len(evals)
        with open(overallFile, 'a+') as ro:
            ro.writelines(['%d\t%f\t%f\n' % (f_i, mean_final_error, mean_abs)])

        # for i in range(full_train.shape[0]):
        MSE += (got_it[0, -1]*normalizer - evals[-1]) ** 2

        print('MSE after file %d : %f' % (count, (MSE/count)))

MSE /= count
print('Final MSE: %f' % MSE)
