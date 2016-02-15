from numpy.linalg import LinAlgError
import theano
import theano.tensor as T
import os
import sys
import statsmodels.api as sm
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample

__author__ = 'idurugkar'

start_i = 100
end_i = 160000

num_units = 1
num_inputs = 1
num_outputs = 1

input_var = T.matrix(name='inputs', dtype=theano.config.floatX)
target_var = T.matrix(name='targets', dtype=theano.config.floatX)



mseFile = 'Data3/mse_st_%d_%d.txt' % (start_i, end_i)
with open(mseFile, 'w') as rf:
    rf.writelines(['Point\tMSE\n'])
absFile = 'Data3/absolute_st_%d_%d.txt' % (start_i, end_i)
with open(absFile, 'w') as ro:
    ro.writelines(['Point\tAbs\n'])


with open('Data3/ST-IWRs.txt', 'r') as fr:
    vals = [float(x) for x in fr.read().split()]
num_vals = len(vals)
MSE = 0.
count = 0
jumps = num_vals/100
# dataset = []

for j in range(start_i/100, end_i/100):
    dataset = vals[:j*100]

    try:
        res = sm.tsa.ARMA(dataset, (3, 2)).fit(trend="nc")
    except (ValueError, LinAlgError):
        print "ERROR"
        pass
    except:
        pass
    # get what you need for predicting one-step ahead
    params = res.params
    residuals = res.resid
    p = res.k_ar
    q = res.k_ma
    k_exog = res.k_exog
    k_trend = res.k_trend
    steps = 1

    try:
        prediction = _arma_predict_out_of_sample(params,
                                                 steps,
                                                 residuals,
                                                 p, q, k_trend,
                                                 k_exog,
                                                 endog=dataset,
                                                 exog=None,
                                                 start=len(dataset))
    except:
        pass
    count += 1

    print('Point %d prediction = %f' % (j*100, prediction))
    print('Target: %f' % vals[j*100])
    with open(mseFile, 'a+') as rf:
        rf.writelines(['%d\t%f\n' % (j*100, (prediction - vals[j*100]) ** 2)])
    with open(absFile, 'a+') as rf:
        rf.writelines(['%d\t%f\n' % (j*100, (prediction - vals[j*100]))])

    MSE += (prediction - vals[j*100]) ** 2
    print('MSE after item %d : %f' % (count, (MSE/count)))

MSE /= count
print('Final MSE: %f' % MSE)
