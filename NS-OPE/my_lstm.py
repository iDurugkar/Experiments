__author__ = 'idurugkar'
import numpy as np
import theano
import theano.tensor as T
from math import sqrt

from lasagne.layers import *
import lasagne
from lasagne import nonlinearities


import matplotlib.pyplot as plt
import seaborn as sns

with open('Data2/IWRs/IWRs_out22.txt', 'r') as fp:
    text = fp.read()
dataset = [float(tok) for tok in text.split()]

evals = []
ind = []

with open('Data2/trueEval.txt', 'r') as fr:
    fr.readline()
    for line in fr.readlines():
        parts = line.split()
        ind.append(int(parts[0]))
        evals.append(float(parts[-1]))
    # evals = [float(line.split()[-1]) for line in fr.readlines()]
sequence_size = 2000

dataset = np.asarray(dataset, dtype=theano.config.floatX)
normalizer = dataset.min()
dataset /= normalizer
# plt.plot(dataset)
# plt.show()
full_data = dataset.reshape((1,-1))
full_train = np.copy(full_data[:, :-1])
full_targets = np.copy(full_data[:, 1:])

dataset = dataset.reshape((-1, sequence_size), order='F')
# dataset = dataset.reshape((1, -1))

num_examples, sequence_size = dataset.shape

num_train = dataset[:].shape[0]
# num_test = dataset[:2].shape[0]

train = np.copy(dataset[:, :-1]) #  theano.shared(np.asarray(dataset[:-10, :], dtype=theano.config.floatX))
                      #borrow=True)
train_targets = np.copy(dataset[:, 1:])
# train_targets = np.zeros_like(dataset[:-10])
# train_targets[:, 1:] = dataset[:-10, :-1]
# train_targets = theano.shared(np.asarray(train_targets, dtype=theano.config.floatX),
#                              borrow=True)

# test = np.copy(dataset[:2, :-1]) #  theano.shared(np.asarray(dataset[-10:], dtype=theano.config.floatX))
                     #borrow=True)
# test_targets = np.copy(dataset[:2, 1:])
# test_targets = np.zeros_like(dataset[-10:])
# test_targets[:, 1:] = dataset[-10:, :-1]
# test_targets = theano.shared(np.asarray(test_targets, dtype=theano.config.floatX),
#                               borrow=True)

num_units = 5
num_inputs = 1
num_outputs = 1

input_var = T.matrix(name='inputs', dtype=theano.config.floatX)
target_var = T.matrix(name='targets', dtype=theano.config.floatX)


def build_net(inputVar):
    # Batch size=None, Sequence_Length=None, number of inputs = num_inputs
    l_inp = InputLayer((None, sequence_size), input_var=inputVar)  # ((None, None, 1))
    # retrieve symbolic references to batch_size and sequence_length
    batch_size, sequence_length = l_inp.input_var.shape
    l_rshp1 = ReshapeLayer(l_inp, (batch_size, sequence_length, 1))

    # retrieve symbolic references to batch_size and sequence_length
    # batch_size, sequence_length, _ = l_rshp1.output_shape

    # The LSTM layer!
    # forget_gate = Gate(b=lasagne.init.Constant(0.3))
    # l_lstm = LSTMLayer(l_rshp1, num_units=num_units)  # , forgetgate=forget_gate)

    # Basic RNN
    # l_lstm = RecurrentLayer(l_rshp1, num_units=num_outputs, nonlinearity=nonlinearities.tanh) # num_units

    # GRU layer
    l_lstm = GRULayer(l_rshp1, num_units=num_units, resetgate=0)
    # Flatten output of batch and sequence so that each time step
    # of each sequence is processed independently.
    # Didn't understand this part :/
    l_rshp2 = ReshapeLayer(l_lstm, (-1, num_units))
    l_dense = DenseLayer(l_rshp2, num_units=num_outputs, nonlinearity=None)
    l_out = ReshapeLayer(l_dense, (batch_size, sequence_length))

    return l_out, l_lstm

network, lstm = build_net(input_var)

# temp = train[1:2]

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(prediction, target_var)
reg = lasagne.objectives.squared_error(prediction, input_var)
reg = reg.mean() * 0.01
loss = loss.mean() + reg


params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.adagrad(loss, params, learning_rate=0.1)
# updates = lasagne.updates.adadelta(loss, params, learning_rate=0.1, rho=0.99)


# generate_output = get_output(network, deterministic=True)
test_prediction = get_output(network, deterministic=True)
test_loss = lasagne.objectives.squared_error(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()

# test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
#                   dtype=theano.config.floatX)

index = T.scalar(name='index', dtype='int64')
# train_fn = theano.function(inputs=[index], outputs=loss,
#                            givens={input_var: train[index],
#                                    target_var: train_targets[index]},
#                            updates=updates)
train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
val_fn = theano.function([input_var, target_var], test_loss, allow_input_downcast=True)

num_epochs = 300
for epoch in range(num_epochs):
    # err = np.zeros((1, num_train), dtype=theano.config.floatX)
    # for i in range(num_train):
    err = train_fn(full_train, full_targets)
    print('Epoch %d, Error %f' % (epoch, err))

# pred_err = val_fn(test, test_targets)
# print('Test error: %f' % pred_err)


out = get_output(network)
get_it = theano.function([input_var], out, allow_input_downcast=True)
# with open('results.txt', 'w') as fw:
#     for i in range(num_train):
#         got_it = get_it(np.reshape(train[i], (1, train[i].shape[0])))
#         fw.writelines([str((i+1)*sequence_size) + ' ' + str(got_it[0, -1] * normalizer) + '\n'])
#     for i in range(num_test):
#         got_it = get_it(np.reshape(test[i], (1, test[i].shape[0])))
#         fw.writelines([str((num_train+i+1)*sequence_size) + ' ' + str(got_it[0, -1] * normalizer) + '\n'])
#

# dataset = dataset.reshape((1, -1))

final_error = 0.
abs_error = 0.
got_it = get_it(full_train)
with open('Data2/results_random.txt', 'w') as fw:
    # for i in range(full_train.shape[1]):
    #     if i % 500 == 0:
    for i in range(1, len(ind)):
        pred = (got_it[0, ind[i]-1] * normalizer)
        error = (pred - evals[i])
        final_error += error ** 2
        abs_error += abs(error)
        fw.writelines([str(ind[i]) + ' ' + str(pred) + '\n'])
mean_final_error = final_error / len(evals)
mean_abs = abs_error / len(evals)


print('root mean squared error = %f' % sqrt(mean_final_error))
print('Absolute error = %f' % mean_abs)
# plt.plot(full_train[0])
# plt.plot(got_it[0], linestyle='--', color='r')
# plt.show()

# plt.clf()
# plt.plot(got_it[0]*normalizer, color='r')
# plt.plot(ind, evals, linestyle='-', color='b')
# plt.show()
# plt.savefig('mars_10_500_nd.png')
