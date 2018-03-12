# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 02:59:20 2018

@author: Simran Lamba
"""


import os
os.chdir('C:/Users/lamba_s/Projects/ML')
import pandas as pd
import numpy as np
from time import time
import nltk

import sklearn.feature_extraction.text as tx
from random import shuffle
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack
from sklearn.utils import shuffle



# TF from sklearn
def get_tfmatrix_with_affine(text_train, text_test, ngrams_n = 1, idf=False):

    n_train = np.shape(text_train)[0]
    n_test = np.shape(text_test)[0]
    if not idf:
        if ngrams_n==1:
            print("Extracting unigram features...")
            tf_vectorizer = tx.CountVectorizer()
        elif ngrams_n==2:
            print("Extracting uni+bigram features...")
            tf_vectorizer = tx.CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b')
        elif ngrams_n==3:
            print("Extracting uni+bi+trigram features...")
            tf_vectorizer = tx.CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b')
        else:
            raise('Wrong input for ngrams_n')

    else:
        print("Extracting tf-idf features...")
        tf_vectorizer = tx.TfidfVectorizer(norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False)

    t0 = time()
    tf_vectorizer.fit(text_train)
    features_names = tf_vectorizer.get_feature_names()

    tf = tf_vectorizer.transform(text_train)
    bias_train = csr_matrix(np.transpose(np.matrix(np.ones(n_train))))
    tf_with_affine = csr_matrix(hstack((bias_train, tf)))

    tf_test = tf_vectorizer.transform(text_test)
    bias_test = csr_matrix(np.transpose(np.matrix(np.ones(n_test))))
    tf_with_affine_test = csr_matrix(hstack((bias_test, tf_test)))

    print("done in %0.3fs." % (time() - t0))
    return tf_with_affine, tf_with_affine_test, features_names




#######################################################
# Shuffle
def shuffle_trainset(X_train, y_train):
    tied = list(zip(X_train, y_train))
    shuffle(tied)
    X_train, y_train = zip(*tied)
    return vstack(X_train), y_train



# Read
#n_samples = 500000
#print(n_samples)
train = pd.read_csv('reviews_tr.csv')#, nrows = n_samples)
test = pd.read_csv('reviews_te.csv')#,nrows = np.floor(n_samples/3))
text_train = train['text']
text_test = test['text']
n = len(train)
n_test = len(test)

#hv = HashingVectorizer()
#text_train=hv.transform(text_train)

## Online Perceptron
X_train, X_test, feature_names = get_tfmatrix_with_affine(text_train, text_test, ngrams_n = 1, idf=False)
feature_names_with_bias = np.append(['bias'], feature_names)
y_train = train['label']
y_test = test['label']
pd.options.mode.chained_assignment = None  # default='warn'
y_train[y_train==0] = -1
y_test[y_test==0] = -1



# Shuffle
print("Shuffling and iterating first pass...")
t0 = time()
X_train, y_train = shuffle_trainset(X_train, y_train)
#X_train, y_train = shuffle(X_train, y_train, random_state=0)

w = np.matlib.zeros(np.shape(X_train[0]))
k=0
for i in range(n):
    x_arr = X_train[i]
    y = np.matrix(y_train[i])
    x = x_arr
    w_t = np.transpose(w)
    if y*(x*w_t) <= 0:
        k += 1
        w = w + y*x

print("done in %0.3fs." % (time() - t0))

# Shuffle again
print("Shuffling and iterating second pass...")
t0 = time()
X_train, y_train = shuffle_trainset(X_train, y_train)
#X_train, y_train = shuffle(X_train, y_train, random_state=10)

w_list = [[w,1]]
k=0
for i in range(n):
    x_arr = X_train[i]
    y = np.matrix(y_train[i])
    x = x_arr
    w_t = np.transpose(w)
    if y*(x*w_t) <= 0:
        k += 1
        w = w + y*x
        w_list.append([w,1])

    else:
        w_list[k][1] += 1


w_final = sum([t[0]*t[1] for t in w_list])/(n+1)
print(w_final)
w_final_col = np.transpose(w_final)
w_arr = np.array(w_final)[0]

print("done in %0.3fs." % (time() - t0))


# Train error
y_col = np.transpose(np.matrix(y_train))
values = np.multiply(y_col,X_train*w_final_col)
error_matrix = sum(values<0)/n
error_rate = error_matrix[0,0]
print('Train Error rate: %0.3f' % (error_rate*100), '%')


# Test error
y_col_te = np.transpose(np.matrix(y_test))
values_te = np.multiply(y_col_te,X_test*w_final_col)
error_matrix_te = sum(values_te<0)/n_test
error_rate_te = error_matrix_te[0,0]
print("Test Error rate: %0.3f" % (error_rate_te*100), '%')


max_indices = w_arr.argsort()[-10:][::-1]
min_indices = w_arr.argsort()[:10]
highest_weights = w_arr[max_indices]
lowest_weights = w_arr[min_indices]
words_with_highest_weights = feature_names_with_bias[max_indices]
words_with_lowest_weights = feature_names_with_bias[min_indices]

np.shape(w_arr.argsort())
np.shape(w_arr.argsort()[-3:])
