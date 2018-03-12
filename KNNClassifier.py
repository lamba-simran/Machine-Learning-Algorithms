# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:59:19 2018

@author: Simran Lamba
"""
import math

from scipy.io import loadmat
ocr = loadmat('C:/Users/lamba_s/Documents/Graduate School/Semester 2/Machine Learning for Data Science/ocr.mat')


## view sample image from dataset
import matplotlib.pyplot as plt
from matplotlib import cm
plt.imshow(ocr['data'][0].reshape((28,28)), cmap=cm.gray_r)
plt.show()




import numpy as np


## load data into arrays and split into train and test
data = np.asarray(ocr['data'])
print (data.shape)

labels = np.asarray(ocr['labels'])
print (labels.shape)

train = data[0:42000,:]
train_labels = labels[0:42000]

test = data[42000:,:]
test_labels = labels[42000:]

print (train.shape)
print (test.shape)




k = 5
from scipy import stats

test = ocr['testdata']
test_labels = ocr['testlabels']

## array for predictions on test set
pred = np.zeros((len(test_labels)))

import random

## lists for errors recorded with different values of n
n1000 = []
n2000 = []
n4000 = []
n8000 = []

## ten iterations of random process
for run in range(10):
    print (run)
    ## n, which is size of training set, is tried at different values
    for n in [1000, 2000, 4000, 8000]:
        print (n)
        ## select n random values from training set
        sel = random.sample(xrange(60000), n)
        train = ocr['data'][sel].astype('float')
        train_labels = ocr['labels'][sel]
        ## iterate one by one through test set
        for i, point in enumerate(test):
            ## mechanics below explained in write-up. it performs nearest neighbor computation
            copied_points = np.tile(point, (train.shape[0], 1))
            differences = copied_points - train
            distances = np.linalg.norm((differences), axis=1)
            indices = np.argsort(distances)[:k]
            voted_labels = []
            for j in indices:
                voted_labels.append(train_labels[j])
            pred[i] = stats.mode(voted_labels)[0][0]
        ## calculate error rate for current run
        errors = 0
        for i in range(len(pred)):
            if pred[i] != test_labels[i]:
                errors += 1
        error_rate = float(errors)/len(pred)
        print (error_rate)
        ## append error rate to proper list
        if n == 1000:
            n1000.append(error_rate)
        elif n == 2000:
            n2000.append(error_rate)
        elif n == 4000:
            n4000.append(error_rate)
        elif n == 8000:
            n8000.append(error_rate)

## calculate means of error rate at different n
n1000_a = np.mean(n1000)
n2000_a = np.mean(n2000)
n4000_a = np.mean(n4000)
n8000_a = np.mean(n8000)

## calculate standard deviation of error rate at different n
n1000_s = np.std(n1000)
n2000_s = np.std(n2000)
n4000_s = np.std(n4000)
n8000_s = np.std(n8000)


## plot means of error rates at different n with standard deviation error bars
plt.figure(figsize=(16,8))
plt.title('Learning Curve')
plt.ylabel('test error')
plt.xlabel('size of training set')
plt.errorbar(
    [1000,2000,4000,8000],  # X
    [n1000_a,n2000_a,n4000_a,n8000_a], # Y
    yerr=[n1000_s, n2000_s, n4000_s, n8000_s],     # Y-errors
    fmt="rs--", # format line like for plot()
    ecolor='blue',
    linewidth=1,	# width of plot line
    elinewidth=5
    )
plt.show()
