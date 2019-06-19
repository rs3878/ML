
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### read data and labels

# In[2]:


X = pd.read_csv("X.csv", header = None)
y = pd.read_csv("y.csv", header = None)
data = X.copy()
data['y'] = y
data = data.values
d = data.shape[1]-1

# labels
with open('README') as f:
    file=f.read()
file = file.split('\n') 
file = file[2:] 
# labels include both feature values and its index
labels = []
# legend is solely a list of feature values
legend = []
for i in file:
    s = i.split(" ")
    if int(s[0]) <= 9:
        labels.append([s[0],s[2]])
        legend.append(s[2])
    else:
        labels.append([s[0],s[1]])
        legend.append(s[1])


# #### split data for naive bayesian classifier and knn

# In[3]:


def split_data(X, test = False):
    # when we split training data
    if test == False:
        X0 = X[X[:,-1]==0]
        X1 = X[X[:,-1]==1]

        y = X[:,-1]
        y0 = y[y==0]
        y1 = y[y==1]

        X = X[:,:-1]
        X0 = X0[:,:-1]
        X1 = X1[:,:-1]

        n,d = X.shape
        n0,n1 = X0.shape[0],X1.shape[0]
        
        return X,X0,X1,y0,y1,n,d,n0,n1
    
    # when we split test data
    if test == True:
        y = X[:,-1]
        X = X[:,:-1]
        n_test = X.shape[0]
        
        return n_test, X, y


# #### define params

# In[4]:


def find_pi(y1,y0):
    # find the bernoulli param
    pi = len(y1)/(len(y1)+len(y0))
    return pi


# In[5]:


def find_lam(X0,X1,d,n0,n1):
    # find the poisson param for two y categories (y=0 and y=1)
    lam_0 = np.ones(d)
    for d_index in range(d):
        numerator = np.sum(X0[:,d_index]) +1
        denominator = n0 +1
        lam_0[d_index] = numerator/denominator

    lam_1 = np.ones(d)
    for d_index in range(d):
        numerator = np.sum(X1[:,d_index]) +1
        denominator = n1 +1
        lam_1[d_index] = numerator/denominator
    
    return lam_0,lam_1


# #### find probability of falling in y=1 or y=0 category

# In[6]:


from scipy.stats import poisson
from scipy.stats import bernoulli

def find_prob(x_test, pi,lam_0, lam_1, d, n):
    # use the bernoulli param
    bern_0 = bernoulli.logpmf(k=0,p=pi)
    bern_1 = bernoulli.logpmf(k=1,p=pi)
    
    # use the poisson param
    poisson_0, poisson_1 = 0, 0
    for i in range(d):
        poi_0 = poisson.logpmf(x_test[i],lam_0[i])
        poi_1 = poisson.logpmf(x_test[i],lam_1[i])
        poisson_0 += poi_0
        poisson_1 += poi_1
        
    # the final ans
    result_0 = bern_0 + poisson_0
    result_1 = bern_1 + poisson_1
    
    return result_0, result_1


# In[7]:


def get_pred(prob_0,prob_1):
    prob = [prob_0,prob_1]
    return np.argmax(prob,axis = 0)


# In[8]:


def find_pred(y1,y0,X0,X1,d,n0,n1,n,X_test):
    # get the params
    pi = find_pi(y1,y0)
    lam_0, lam_1 = find_lam(X0,X1,d,n0,n1)
    
    # calculate the prob of belonging to y=1 or y=0 case for each subject
    prob_0, prob_1  = np.ones(n), np.ones(n)
    for i in range(n):
        prob_0[i], prob_1[i] = find_prob(X_test[i],pi,lam_0,lam_1,d,n)
        
    # make a prediction based on the prob 
    pred = get_pred(prob_0,prob_1)
    return  pred, lam_0, lam_1


# #### shuffle and partition

# In[9]:


def train_test_split(data, i, n):
    # test data
    test = data[i]
    
    # train data : stack the rest of 9 partitions 
    if i == 0:
        train = np.vstack(data[1:])
    elif i == n-1:
        train = np.vstack(data[:n-1])
    else:
        pt1 = np.vstack(data[:i])
        pt2 = np.vstack(data[i+1:])
        train = np.vstack([pt1,pt2])
    return train, test


# ## Naive Bayesian Classifier

# In[10]:


from sklearn.metrics import confusion_matrix 
result = np.zeros((2,2))
times = 10
np.random.shuffle(data)
splited = np.split(data,times)
lambda_0, lambda_1 = np.ones(d),np.ones(d)

for i in range(times):
    x_train, x_test = train_test_split(splited, i, times)
    X,X0,X1,y0,y1,n,d,n0,n1 = split_data(x_train)
    n_test, X_test, y_test = split_data(x_test, test = True)
    
    # calculated parameters to make the predictions
    pred, lam_0, lam_1 = find_pred(y1,y0,X0,X1,d,n0,n1,n_test,X_test)
    lambda_0 += lam_0
    lambda_1 += lam_1
    
    # confusion matrix for assessing accuracy
    matrix = confusion_matrix(y_test, pred)
    for i in range(2):
        for j in range(2):
            result[i][j] += matrix[i][j]
            
lambda_0 /=10
lambda_1 /=10
result


# In[11]:


accuracy = (result[0][0]+result[1][1])/4600
accuracy


# In[103]:


#stem plot
plt.stem(lambda_0, 'b', markerfmt = 'bo', label ='lambda_0')
plt.stem(lambda_1, 'r', markerfmt = 'ro', label ='lambda_1')
plt.xticks(rotation = 40)
plt.legend()


# In[14]:


legend[15],legend[51]


# ### Obeservation:

# dimension 16 and dimension 52 are 'free' and '!'. They both seem to be popular in the spam emails than compare to the non-spam emails, since their $\lambda_1$s are much larger than their $\lambda_0$s. Empirical evidence helps us understand this observation: there are a lot of "free" stuff offered in the spam emails and in order to show enthusiasm, spam emails use "!" quite often.

# ## KNN

# #### if doing a weighted majority vote taking consideration of the sum of the distances, which yields a batter result

# In[15]:


result = np.zeros((2,2))
times = 10
np.random.shuffle(data)
splited = np.split(data,times)
accuracy_list = []

# k = 1 to 20
for k in range(1,21):
    nbr = int((k+1)/2)
    
    # partition : 10 train_test split
    for index in range(times):
        x_train, x_test = train_test_split(splited, index, times)
        X,X0,X1,y0,y1,n,d,n0,n1 = split_data(x_train)
        n_test, X_test, y_test = split_data(x_test, test = True)
        pred = np.ones(n_test)
        
        # compare for each test subjects
        for j in range(len(X_test)):
            if_0 = np.sort(np.sum(np.abs(X0-X_test[j]),axis = 1))
            if_1 = np.sort(np.sum(np.abs(X1-X_test[j]),axis = 1))
            if np.sum(if_0[:nbr]) < np.sum(if_1[:nbr]):
                pred[j] = 0
            elif np.sum(if_0[:nbr]) > np.sum(if_1[:nbr]):
                pred[j] = 1
            # if they have the same distance sum, make an educated guess
            else:
                if type(k/2) != int:
                    if if_0[nbr-1] > if_1[nbr-1]:
                        pred[j] = 0
                    elif if_0[nbr-1] < if_1[nbr-1]:
                        pred[j] = 1
                    else:
                        pred[j] = np.random.binomial(n=1,p=0.5)
                else:
                    pred[j] = np.random.binomial(n=1,p=0.5)
        
        # confusion matrix for assessing accuracy
        matrix = confusion_matrix(y_test, pred)
        for i in range(2):
            for j in range(2):
                result[i][j] += matrix[i][j]
                
    accuracy = (result[0][0]+result[1][1])/4600
    result = np.zeros((2,2))
    accuracy_list.append(accuracy)


# In[16]:


accuracy_list


# In[17]:


plt.plot([i for i in range(1,21)], accuracy_list)


# #### this version takes an unweighted majority vote, which is just the version we learned in class, accuracy is slightly worse

# In[18]:


result = np.zeros((2,2))
times = 10
np.random.shuffle(data)
splited = np.split(data,times)
accuracy_list = []

# k = 1 to 20
for k in range(1,21):
    nbr = int((k+1)/2)
    
    # partition : 10 train_test split
    for index in range(times):
        x_train, x_test = train_test_split(splited, index, times)
        X,X0,X1,y0,y1,n,d,n0,n1 = split_data(x_train)
        n_test, X_test, y_test = split_data(x_test, test = True)
        pred = np.ones(n_test)
        
        # compare for each test subjects
        for j in range(len(X_test)):
            if_0 = np.sort(np.sum(np.abs(X0-X_test[j]),axis = 1))
            if_1 = np.sort(np.sum(np.abs(X1-X_test[j]),axis = 1))
            if if_0[nbr] < if_1[nbr]:
                pred[j] = 0
            elif if_0[nbr] > if_1[nbr]:
                pred[j] = 1
            # if they have the same distance sum, make an educated guess
            else:
                if np.sum(if_0[:nbr])< np.sum(if_1[:nbr]):
                        pred[j] = 0
                elif np.sum(if_0[:nbr])> np.sum(if_1[:nbr]):
                        pred[j] = 1
                elif type(k/2) != int:
                    if if_0[nbr-1] < if_1[nbr-1]:
                        pred[j] = 0
                    elif if_0[nbr-1] > if_1[nbr-1]:
                        pred[j] = 1
                    elif np.sum(if_0[:nbr])< np.sum(if_1[:nbr]):
                        pred[j] = np.random.binomial(n=1,p=0.5)
                else:
                    pred[j] = np.random.binomial(n=1,p=0.5)
        
        # confusion matrix for assessing accuracy
        matrix = confusion_matrix(y_test, pred)
        for i in range(2):
            for j in range(2):
                result[i][j] += matrix[i][j]
                
    accuracy = (result[0][0]+result[1][1])/4600
    result = np.zeros((2,2))
    accuracy_list.append(accuracy)


# In[19]:


accuracy_list


# In[20]:


plt.plot([i for i in range(1,21)], accuracy_list)


# #### this version uses cdist to calculate the distance between two matrices, which is the fastest

# In[21]:


import scipy.spatial.distance as dist
result = np.zeros((2,2))
times = 10
np.random.shuffle(data)
splited = np.split(data,times)
accuracy_list = []

# k = 1 to 20
for k in range(1,21):
    nbr = int((k+1)/2)
    
    # partition : 10 train_test split
    for index in range(times):
        x_train, x_test = train_test_split(splited, index, times)
        X,X0,X1,y0,y1,n,d,n0,n1 = split_data(x_train)
        n_test, X_test, y_test = split_data(x_test, test = True)
        pred = np.ones(n_test)
        
        #Y0 = dist.cdist(X0, X_test, 'minkowski', p=1)
        #Y1 = dist.cdist(X1, X_test, 'minkowski', p=1)
        Y0 = dist.cdist(X0, X_test, 'cityblock')
        Y1 = dist.cdist(X1, X_test, 'cityblock')
        
        for i in range(len(X_test)):
            y0 = np.sort(Y0[:,i])[:nbr]
            y1 = np.sort(Y1[:,i])[:nbr]
            if y0[-1] < y1[-1]:
                pred[i] = 0
            elif y0[-1] > y1[-1]:
                pred[i] = 1
            else:
                pred[i] = np.random.binomial(n=1,p=0.5)
        
        # confusion matrix for assessing accuracy
        matrix = confusion_matrix(y_test, pred)
        for i in range(2):
            for j in range(2):
                result[i][j] += matrix[i][j]
                
    accuracy = (result[0][0]+result[1][1])/4600
    result = np.zeros((2,2))
    accuracy_list.append(accuracy)


# In[22]:


accuracy_list


# In[23]:


plt.plot([i for i in range(1,21)], accuracy_list)


# ## Logistic Regression

# In[24]:


X = pd.read_csv("X.csv", header = None)
y = pd.read_csv("y.csv", header = None)
X_logit = X.copy()
y_logit = y.copy()
y_logit[y_logit==0] = -1
X_logit['intercept'] = 1
X_logit['y_logit'] = y_logit
X_logit = X_logit.values
y_logit = y_logit.values


# ### Steepest Ascent

# #### This version uses for loops, it's slightly slower.

# In[27]:


def sigmoid(x,y,w):
    return np.exp(y*(x@w))/(1+np.exp(y*(x@w)))


# In[28]:


times = 10
np.random.shuffle(X_logit)
splited = np.split(X_logit,times)
obj_fcn = []
step_size = 0.01/4600
iteration = 1000

for i in range(times):
    x_train, x_test = train_test_split(splited, i, times)
    y_test = x_test[:,-1]
    y_train = x_train[:,-1]
    x_test = x_test[:,:-1]
    x_train = x_train[:,:-1]
    n,d = x_train.shape
    
    w = np.zeros(d)
    likelihood_list = np.zeros(iteration)
    for t in range(iteration):
        change = 0
        likelihood = 0
        for j in range(n):
            sig = sigmoid(x_train[j],y_train[j],w)
            likelihood += np.log(sig)
            change += (1-sig)*y_train[j]*x_train[j]
        w = w + step_size*change
        likelihood_list[t] = likelihood
    obj_fcn.append(likelihood_list)


# In[29]:


# plot for all 10 
for index in range(times):
    plt.plot([i for i in range(iteration)], obj_fcn[index])


# #### Here is the vectorized version

# In[60]:


times = 10
np.random.shuffle(X_logit)
splited = np.split(X_logit,times)
obj_fcn = []
step_size = 0.01/4600
iteration = 1000
result = np.zeros((2,2))

for i in range(times):
    x_train, x_test = train_test_split(splited, i, times)
    y_test = x_test[:,-1]
    y_train = x_train[:,-1]
    x_test = x_test[:,:-1]
    x_train = x_train[:,:-1]
    n,d = x_train.shape
    
    w = np.zeros(d)
    likelihood_list = np.zeros(iteration)
    for t in range(iteration):
        sigm = np.exp(y_train*(x_train@w))/(1+np.exp(y_train*(x_train@w)))
        likelihood = np.sum(np.log(sigm))
        step1 = (x_train.T*y_train).T
        step2 = (step1.T)*(1-sigm)
        change = np.sum(step2.T, axis =0)
        w = w + step_size*change
        likelihood_list[t] = likelihood
    pred = x_test@w
    pred[pred<0.5]=-1
    pred[pred>=0.5]=1
    matrix = confusion_matrix(y_test, pred)
    for i in range(2):
        for j in range(2):
            result[i][j] += matrix[i][j]
    obj_fcn.append(likelihood_list)
                
accuracy = (result[0][0]+result[1][1])/4600
accuracy


# In[52]:


# plot for all 10 
for index in range(times):
    plt.plot([i for i in range(iteration)], obj_fcn[index])


# ### Newton's Method

# In[91]:


iteration = 100
result = np.zeros((2,2))

times = 10
np.random.shuffle(X_logit)
splited = np.split(X_logit,times)
obj_fcn = []
result = np.zeros((2,2))

for i in range(times):    
    x_train, x_test = train_test_split(splited, i, times)
    y_test = x_test[:,-1]
    y_train = x_train[:,-1]
    x_test = x_test[:,:-1]
    x_train = x_train[:,:-1]
    n,d = x_train.shape
    w = [10**(-12) for i in range(55)]
    likelihood_list = np.zeros(iteration)

    for t in range(iteration):
        sigm = np.exp(y_train*(x_train@w))/(1+np.exp(y_train*(x_train@w)))
        likelihood = np.sum(np.log(sigm))
        likelihood_list[t] = likelihood
        dw = x_train.T@(y_train*(1-sigm))
        ddw = - x_train.T@(np.diag(sigm*(1-sigm)))@x_train
        #dw = (x_train.T@(1/(1+np.exp(y_train*(x_train@w)))*y_train))
        #ddw = -x_train.T@(np.diag(sigm*(1-sigm)))@x_train
        w = w - (np.linalg.inv(ddw))@dw
        
    obj_fcn.append(likelihood_list)
    
    # confusion matrix for assessing accuracy
    pred = x_test@w
    pred[pred<0.5] = -1
    pred[pred>=0.5] = 1
    matrix = confusion_matrix(y_test, pred)
    for i in range(2):
        for j in range(2):
            result[i][j] += matrix[i][j]
                
accuracy = (result[0][0]+result[1][1])/4600
accuracy


# In[92]:


# plot for all 10 
for index in range(times):
    plt.plot([i for i in range(iteration)], obj_fcn[index])


# In[93]:


result

