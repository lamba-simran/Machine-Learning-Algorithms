# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:58:12 2018

@author: Simran Lamba
"""

import scipy.io as spio
import numpy as np
import scipy.optimize as opt
import scipy.stats as st
import statsmodels.api as sm
import itertools as it
import os
import matplotlib.pyplot as plt
import pandas as pd


os.chdir("C:\\Users\\lamba_s\\Projects\\ML")

mat = spio.loadmat('wine.mat', squeeze_me=True)
x = mat['data']
y = mat['labels']

### statsmodels.api
olsmodel = sm.OLS(y, x)
results = olsmodel.fit()
results.summary()
theta = results.params

## Normal equation: theta = (X'X)^-1.X'y
X = np.matrix(x)
Y = np.transpose(np.matrix(y))
X_t = np.transpose(X)

# Check if estimated parameters satisfy the normal equations
theta_est = np.transpose(np.matrix(theta))
theta_calc = np.linalg.inv(X_t*X)*X_t*Y
theta_est - theta_calc < 1e-10

## Test
test_x = mat['testdata']
test_y = mat['testlabels']
y_pred = olsmodel.predict(params = theta, exog=test_x)

### Scipy
olsmodel2 = opt.lsq_linear(x, y,  lsq_solver = 'exact')

test_mse = sum([p**2 for p in (test_y - y_pred)])/len(test_y)
print('Test Risk of OLS:', test_mse)

# MSE of train data
train_y_pred = olsmodel.predict(params = theta, exog=x)
train_mse = sum([p**2 for p in (y - train_y_pred)])/len(y)

#####################################################
######### Sparse Linear Predictor ###################
#####################################################

combs = it.combinations(range(1,12), 3)

lb = np.array([-np.finfo(float).eps]*12)
ub = np.array([np.finfo(float).eps]*12)

#slp = opt.lsq_linear(x, y, bounds = (lb,ub), lsq_solver = 'exact')
n = len(x)
costs = []
features = []

for c in combs:
    lb = np.array([-np.finfo(float).eps]*12)
    ub = np.array([np.finfo(float).eps]*12)
    lb[0] = -np.inf
    ub[0] = np.inf
    lb[list(c)] = -np.inf
    ub[list(c)] = np.inf
    slp = opt.lsq_linear(x, y, bounds = (lb,ub), lsq_solver = 'exact')
    costs.append(slp.cost/n*2)
    features.append(c)

min_ind = costs.index(min(costs))
min_cost = min(costs)

c = features[min_ind]
lb = np.array([-np.finfo(float).eps]*12)
ub = np.array([np.finfo(float).eps]*12)
lb[0] = -np.inf
ub[0] = np.inf
lb[list(c)] = -np.inf
ub[list(c)] = np.inf

slp = opt.lsq_linear(x, y, bounds = (lb,ub), lsq_solver = 'exact')

beta = np.transpose(np.matrix(slp.x))
train_mse_slp = sum([((X*beta - Y)[i,0])**2 for i in range(len(X))])/len(X)

test_X = np.matrix(test_x)
test_Y = np.transpose(np.matrix(test_y))
test_mse_slp = sum([((test_X*beta - test_Y)[i,0])**2 for i in range(len(test_X))])/len(test_X)
print('Test Risk of Sparse Linear Predictor:', test_mse_slp)
print('Feature Indices:', c)
print('Parameters:', beta[[0]+list(c)])

corrs = dict()
for i in c:
    v = test_x[:,i]
    other_vars = set(np.arange(1,12)) - set([i])

    a = []
    for j in other_vars:
        pearson_corr = st.pearsonr(v,test_x[:,j])[0]
        a.append([pearson_corr,j])
    corrs[i] = pd.DataFrame(a)

ans = dict()
for ind in corrs:
    corr_descending = corrs[ind].iloc[np.argsort(abs(corrs[ind][0])).values].iloc[::-1]
    ans[ind] = [list(corr_descending.iloc[0]), list(corr_descending.iloc[1])]

print('Most correlated (format-[Pearson Correlation coeff, feature index]):',ans)
