
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


x_train = pd.read_csv("X_train.csv", header = None)
y_train = pd.read_csv("y_train.csv", header = None)
x_test = pd.read_csv("X_test.csv", header = None)
y_test = pd.read_csv("y_test.csv", header = None)
x_train = x_train.values
y_train = y_train.values
x_test = x_test.values
y_test = y_test.values


# In[38]:


n,d = x_train.shape
identity = np.identity(d)
W = np.ones((5000,d))
df_lam = np.ones(5000)


# In[39]:


# if not using svd
for i in range(5000):
    inv = np.linalg.inv(i*identity+(x_train.T@x_train))
    W[i] = (inv@x_train.T@y_train).reshape(d,)
    df_lam[i] = np.trace(x_train@inv@x_train.T)


# In[78]:


# if using svd
WRR = np.ones((5000,d))
for i in range(5000):
    U, S, VT = np.linalg.svd(x_train, full_matrices = False)
    SS = np.diag(S/(i+S**2))
    a = VT.T@SS@U.T
    WRR[i] = (a @ y_train).reshape(d,)


# In[83]:


features = ['cylinders','displacement','horsepower','weight','acceleration','year made','intercept']
for i in range(d):
    plt.plot(df_lam, WRR[:,i], label = features[i])
plt.legend(loc = 'lower left')
plt.xlabel("degree of freedom of lambda")
plt.ylabel("Wrr: ridge regression weights")
plt.title("Wrr vs. df(lambda)")


# ####  Weight and year made clearly stand out, which makes sense as weight and year made seem to be really important factors that affect miles per gallon a car will get.

# In[6]:


def RMSE(x_train, y_train, x_test, y_test, n_trials = 100):
    RMSE = np.ones(n_trials)
    identity = np.identity(x_train.shape[1])
    for i in range(n_trials):
        inv = np.linalg.inv(i*identity+(x_train.T@x_train))
        w = (inv@x_train.T@y_train)
        y = x_test@w
        RMSE[i] = np.sqrt(np.sum(np.square(y-y_test))/len(y_test))
    return RMSE


# In[7]:


rmse = RMSE(x_train, y_train, x_test, y_test, n_trials = 50)


# In[8]:


plt.plot(rmse)


# #### When lambda equals 0, this problem achieves least RMSE, which turns out to have the best result. Therefore we should choose lambda to be 0  in this case. In other words, no wonder we use least square all the time. Although ridge regression takes in consideration of the variance in trade off of some bias, more often least square has the best result already. 

# In[9]:


rmse_1 = RMSE(x_train, y_train, x_test, y_test)


# In[10]:


def square_standardize(mat):
    means = []
    stds = []
    matrix = np.square(mat[:,0:-1])
    for i in range(matrix.shape[1]):
        means.append(matrix[:,i].mean())
        stds.append(matrix[:,i].std())
        matrix[:,i] = (matrix[:,i] - matrix[:,i].mean())/ matrix[:,i].std()
    return np.hstack((mat, matrix)), means, stds


# In[11]:


def test_square_standardize(mat, means, stds):
    matrix = np.square(mat[:,0:-1])
    for i in range(matrix.shape[1]):
        matrix[:,i] = (matrix[:,i] - means[i])/ stds[i]
    return np.hstack((mat, matrix))


# In[12]:


x_train_2, means_2, stds_2 = square_standardize(x_train)
x_test_2 = test_square_standardize(x_test, means_2, stds_2)
rmse_2 = RMSE(x_train_2, y_train, x_test_2, y_test)


# In[14]:


def thirdpower_standardize(mat):
    matrix = np.power(mat[:,0:6],3)
    mat, m, s = square_standardize(mat)
    means = []
    stds = []
    for i in range(matrix.shape[1]):
        means.append(matrix[:,i].mean())
        stds.append(matrix[:,i].std())
        matrix[:,i] = (matrix[:,i] - matrix[:,i].mean())/ matrix[:,i].std()
    return np.hstack((mat, matrix)), means, stds


# In[15]:


def test_thirdpower_standardize(x_train, mat, means, stds):
    matrix = np.power(mat[:,0:-1],3)
    a, m, s = square_standardize(x_train)
    mat = test_square_standardize(mat, m, s)
    for i in range(matrix.shape[1]):
        matrix[:,i] = (matrix[:,i] - means[i])/ stds[i]
    return np.hstack((mat, matrix))


# In[16]:


x_train_3, means_3, stds_3 = thirdpower_standardize(x_train)
x_test_3 = test_thirdpower_standardize(x_train, x_test, means_3, stds_3)
rmse_3 = RMSE(x_train_3, y_train, x_test_3, y_test)


# In[17]:


plt.plot(rmse_1, label = "p=1")
plt.plot(rmse_2, label = "p=2")
plt.plot(rmse_3, label = "p=3")
plt.legend()
plt.title("RMSE vs. lambda")
plt.xlabel("lambda")
plt.ylabel("RMSE")


# #### Based on this plot, we should choose p = 3 because it achieves minimum RMSE. My assessment of the ideal value of λ changes to around 50 where it achieves  minimal RMSE across all models
