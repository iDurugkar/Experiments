import numpy as np
with open('AdobeData/IWR-CTR-RF-0.7.txt', 'r') as fr:
    vals = [float(x) for x in fr.read().split()]

with open('AdobeData/abs_IWR-CTR-0.7_prediction_120_160.txt', 'r') as fr:
    txt = fr.readline().split('\r')[1:]
    ind = []
    err = []
    for line in txt:
        ind.append(int(line.split()[0]))
        err.append(float(line.split()[1]))
print 'read everything'

fe = open('AdobeData/abs_0.7_corrected.txt', 'w')
fm = open('AdobeData/mse_0.7_corrected.txt', 'w')

mse = []
for i in range(len(err)):
    a = err[i]
    b = vals[ind[i]+1]
    c = vals[ind[i]]
    err[i] = err[i] + vals[ind[i]+1] - vals[ind[i]]
    mse.append(err[i] ** 2)
    fe.writelines(['%d\t%f\n' % (ind[i], err[i])])
    fm.writelines(['%d\t%f\n' % (ind[i], mse[i])])
print np.mean(mse)
print np.mean(err)
