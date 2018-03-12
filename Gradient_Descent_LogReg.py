
# coding: utf-8

# In[521]:


import scipy.io as spio
import numpy as np
import pandas as pd


# In[522]:


data = spio.loadmat('hw3data (1).mat')
#data


# In[523]:


type(data)


# In[524]:


temp_x = data["data"]
temp_x.shape


# In[525]:


y = data["labels"]
y.shape


# In[526]:


x = np.insert(temp_x, 0, 1, axis=1)
x.shape


# In[527]:


x


# ###  Question 3.b)

# In[528]:


def LossFunc(x,y,b):
    loss = np.log(1+np.exp(np.dot(x,b))) - np.multiply(y,np.dot(x,b))
    return np.mean(loss)


# In[529]:


def grad(x,y,b):
    temp = 1/(1+np.exp(-1*(np.dot(x,b)))) - y
    deriv = np.multiply(temp, x)
    return np.mean(deriv, axis=0).reshape((4,1))


# In[530]:


def backtrack_ls(current_loss,x,y,b):
    a = 1
    gradient = grad(x,y,b)
    gradient2= np.dot(gradient.T,gradient)
    while(LossFunc(x,y,b-a*gradient) > current_loss - 0.5*a*gradient2):
        a = a/2

    return a


# In[531]:


b = np.zeros((4,1))
#b


# In[532]:


alpha = 1
count = 0
loss = LossFunc(x,y,b)

while loss>=0.65064:
    current_loss = loss
    alpha = backtrack_ls(current_loss,x,y,b)
    count = count + 1
    b = b - alpha*grad(x,y,b)
    loss = LossFunc(x,y,b)
    #print(str(loss)+ " loss at epoch: " + str(count))



# In[533]:


print("Number of iterations needed:")
count


# ### Question 3.c)

# In[534]:


features = pd.DataFrame(x)
#features


# In[535]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
features.boxplot()
plt.xticks( rotation=30, ha="right")


# In[536]:


features.describe()


# In[537]:


transformed_x = x
transformed_x[:,1] = (transformed_x[:,1])/np.std(x[:,1])
transformed_x[:,2] = (transformed_x[:,2])/np.std(x[:,2])
transformed_x[:,3] = (transformed_x[:,3])/np.std(x[:,3])


# In[538]:


f = pd.DataFrame(transformed_x)


# In[539]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
f.boxplot()
plt.xticks( rotation=30, ha="right")


# In[540]:


b.fill(0)


# In[541]:


alpha = 1
count = 0
loss = LossFunc(transformed_x,y,b)

while loss>=0.65064:
    current_loss = loss
    alpha = backtrack_ls(current_loss,transformed_x,y,b)
    count = count + 1
    b = b - alpha*grad(transformed_x,y,b)
    loss = LossFunc(transformed_x,y,b)
    #print(str(loss)+ " loss at epoch: " + str(count))


# In[542]:


count


# ### Question 3.d)

# In[543]:


temp_x = data["data"]
x = np.insert(temp_x, 0, 1, axis=1)
x


# In[544]:


x_train = x[:int(0.8*4096), :]
x_val = x[int(0.8*4096):, :]

y_train = y[:int(0.8*4096), :]
y_val = y[int(0.8*4096):, :]


# In[545]:


transformed_x_train = transformed_x[:int(0.8*4096), :]
transformed_x_val = transformed_x[int(0.8*4096):, :]


# ### Modified gradient Descent on the original dataset

# In[546]:


b.fill(0)


# In[547]:


def power_2(n):
    return bool(n and not (n&(n-1)))
power_2(17)


# In[548]:


alpha = 1
count = 0
loss = LossFunc(x_train,y_train,b)
loss_val = LossFunc(x_val,y_val,b)
best_err = 100

while True:
    current_loss = loss
    alpha = backtrack_ls(current_loss,x_train,y_train, b)
    count += 1

    b = b - alpha*grad(x_train,y_train,b)

    loss = LossFunc(x_train, y_train,b)

    if power_2(count):
        m = np.dot(x_val,b)>=0
        n = [int(i) for i in m]

        e= [1 if y_val[i,0]!=n[i] else 0 for i in range(len(n)) ]
        error = sum(e)/len(e)

        print(error)

        if error>0.99*best_err and count >=32:
            obj = LossFunc(x_train,y_train,b)
            break

        best_err= min(best_err, error)



# In[549]:


count


# In[550]:


loss


# In[551]:


error


# ### Modified gradient Descent on the linearly tranformed data

# In[552]:


b.fill(0)


# In[553]:


alpha = 1
count = 0
loss = LossFunc(transformed_x_train,y_train,b)
loss_val = LossFunc(transformed_x_val,y_val,b)
best_err = 100

while True:
    current_loss = loss
    alpha = backtrack_ls(current_loss,transformed_x_train,y_train, b)
    count += 1

    b = b - alpha*grad(transformed_x_train,y_train,b)

    loss = LossFunc(transformed_x_train, y_train,b)

    if power_2(count):
        m = np.dot(transformed_x_val,b)>=0
        n = [int(i) for i in m]

        e= [1 if y_val[i,0]!=n[i] else 0 for i in range(len(n)) ]
        error = sum(e)/len(e)

        print(error)

        if error>0.99*best_err and count >=32:
            obj = LossFunc(transformed_x_train,y_train,b)
            break

        best_err= min(best_err, error)


# In[554]:


count


# In[555]:


loss


# In[556]:


error
