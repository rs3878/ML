#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy.stats as st
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# # K-Means

# $$
# \begin{array}{l}{\text { Algorithm: K-means clustering }} \\ {\text { Given: } x_{1}, \ldots, x_{n} \text { where each } x \in \mathbb{R}^{d}} \\ {\text { Goal: Minimize } \mathcal{L}=\sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}\left\{c_{i}=k\right\} || x_{i}-\mu_{k}\left\|^{2}\right.}\end{array}
# $$
# 
# $$
# \begin{array}{c}{\text { Randomly initialize } \boldsymbol{\mu}=\left(\mu_{1}, \ldots, \mu_{K}\right)} \\ {\text { Iterate until } \boldsymbol{c} \text { and } \boldsymbol{\mu} \text { stop changing }} \\ {\text { 1. Update each } c_{i} :} \\ {c_{i}=\arg \min _{k}\left\|x_{i}-\mu_{k}\right\|^{2}}\end{array}
# $$
# 
# $$
# \begin{array}{c}{\text { 2. Update each } \mu_{k} : \text { Set }} \\ {n_{k}=\sum_{i=1}^{n} \mathbb{1}\left\{c_{i}=k\right\} \quad \text { and } \quad \mu_{k}=\frac{1}{n_{k}} \sum_{i=1}^{n} x_{i} \mathbb{1}\left\{c_{i}=k\right\}}\end{array}
# $$

# #### Data Generation

# In[106]:


def generate_X(N = 500):
    mean_1, mean_2, mean_3 = np.array([0,0]), np.array([3,0]), np.array([0,3])
    var_1, var_2, var_3 = np.identity(2),np.identity(2),np.identity(2)
    pi = np.array([0.2,0.5,0.3])
    pt_1 = np.random.multivariate_normal(mean_1, var_1, int(N*pi[0]))
    pt_2 = np.random.multivariate_normal(mean_2, var_2, int(N*pi[1]))
    pt_3 = np.random.multivariate_normal(mean_3, var_3, int(N*pi[2])) 
    X = np.vstack((pt_1,pt_2, pt_3))
    return X.T


# #### Helper Functions

# In[107]:


def calc_centroid(d):
    centroids = []
    for i in range(len(d)):
        centroids.append(np.average(d['cluster_%d' %i ], axis =1))
    return np.array(centroids)


# In[108]:


def calc_dist(centroids, X, d):
    dists = {}
    obj_fcn = 0
    for i in range(centroids.shape[0]):
        dists['cluster_%d' %i ] = np.sum((X.T - centroids[i])**2, axis = 1)
        obj_fcn += np.sum((d['cluster_%d' %i ].T - centroids[i])**2)
    distance = []
    for i in range(len(dists)):
        distance.append(dists['cluster_%d' %i ])
    distance = np.array(distance)
    return distance, obj_fcn


# In[109]:


def new_cluster_assignments(distance, X, d):
    indexes = np.argmin(distance,axis = 0)
    for i in range(len(d)):
        d['cluster_%d' %i ] = X[:,indexes==i]
    return d


# ### Function

# In[110]:


def K_means(X, K, N = 500, iterations = 20):
    # random initialization
    rand_assign = np.random.choice(K, N, p=[1/K for i in range(K)])
    d ={}
    for i in range(K):
        d['cluster_%d' %i ]= X[:,rand_assign==i]
        
    loss = []
    for times in range(iterations):
        centroids = calc_centroid(d)
        distance, obj_fcn = calc_dist(centroids, X, d)
        loss.append(obj_fcn)
        d = new_cluster_assignments(distance, X, d)
    return d, loss


# ### A）For K = 2, 3, 4, 5, plot the value of the K-means objective function per iteration for 20 iterations(the algorithm may converge before that).

# In[111]:


ans = {}
assignment = {}
X = generate_X()
for k in range(2, 6):
    assignment[k], ans[k] = K_means(X, K = k)
    sns.lineplot(range(20), ans[k], label = k)


# In[113]:


ans = {}
assignment = {}
X = generate_X()
for k in range(2, 6):
    assignment[k], ans[k] = K_means(X, K = k, iterations = 21)
    sns.lineplot(range(20), ans[k][1:], label = k)


# ### B) For K = 3, 5, plot the 500 data points and indicate the cluster of each for the final iteration by marking it with a color or a symbol.

# In[38]:


ans = {}
assignment = {}
X = generate_X()
k = 3
assignment[k], ans[k] = K_means(X, K = k)
for j in range(k):
    sns.scatterplot(assignment[k]['cluster_%d' %j ][0], assignment[k]['cluster_%d' %j ][1], label = j)


# In[39]:


ans = {}
assignment = {}
X = generate_X()
k = 5
assignment[k], ans[k] = K_means(X, K = k)
for j in range(k):
    sns.scatterplot(assignment[k]['cluster_%d' %j ][0], assignment[k]['cluster_%d' %j ][1], label = j)


# # Bayes Classifier

# $$
# \begin{array}{l}{\text { Algorithm: Maximum likelihood EM for the GMM }} \\ {\text { Given: } x_{1}, \ldots, x_{n} \text { where } x \in \mathbb{R}^{d}} \\ {\text { Goal: Maximize } \mathcal{L}=\sum_{i=1}^{n} \ln p\left(x_{i} | \pi, \mu, \Sigma\right)}\end{array}
# $$
# 
# $$
# \begin{array}{c}{\text { 1. E-step: For } i=1, \ldots, n, \text { set }} \\ {\phi_{i}(k)=\frac{\pi_{k} N\left(x_{i} | \mu_{k}, \Sigma_{k}\right)}{\sum_{j} \pi_{j} N\left(x_{i} | \mu_{j}, \Sigma_{j}\right)}, \quad \text { for } k=1, \ldots, K}\end{array}
# $$
# 
# $$
# \begin{array}{c}{\text { 2. M-step: For } k=1, \ldots, K, \text { define } n_{k}=\sum_{i=1}^{n} \phi_{i}(k) \text { and update the values }} \\ {\pi_{k}=\frac{n_{k}}{n}, \quad \mu_{k}=\frac{1}{n_{k}} \sum_{i=1}^{n} \phi_{i}(k) x_{i} \quad \Sigma_{k}=\frac{1}{n_{k}} \sum_{i=1}^{n} \phi_{i}(k)\left(x_{i}-\mu_{k}\right)\left(x_{i}-\mu_{k}\right)^{T}}\end{array}
# $$
# 
# $$
# \mathcal{L}=\sum_{i=1}^{n} \sum_{k=1}^{K} \phi_{i}(k) \underbrace{\left\{\ln \pi_{k}+\ln N\left(x_{i} | \mu_{k}, \Sigma_{k}\right)\right\}}_{\ln p\left(x_{i}, c_{i}=k | \pi, \mu_{k}, \Sigma_{k}\right)}+\text { constant w.r.t. } \pi, \boldsymbol{\mu}, \boldsymbol{\Sigma}
# $$

# #### Read Data

# In[42]:


X_train = pd.read_csv("Prob2_Xtrain.csv", header = None).values
y_train = pd.read_csv("Prob2_ytrain.csv", header = None).values
X_test = pd.read_csv("Prob2_Xtest.csv", header = None).values
y_test = pd.read_csv("Prob2_ytest.csv", header = None).values

y_train = y_train.reshape(-1,)
X_train_1 = X_train[y_train==1,:]
X_train_0 = X_train[y_train==0,:]

iterations = 30 # should be 30


# #### Helper Functions

# In[43]:


def E_step(means, covs, pi, X):
    probs = []
    for i in range(len(means)): #for all k
        prob = pi[i]*st.multivariate_normal.pdf(X, mean = means[i], cov = covs[i])
        probs.append(prob)
    probs = np.array(probs).T
    phi = probs/np.sum(probs,axis = 1, keepdims = True)
    return phi


# In[44]:


def loglike(pi, X, means, covs, phi):
    result = 0
    for i in range(len(means)):
        sub_result = st.multivariate_normal.logpdf(X, mean = means[i], cov = covs[i])
        sub_result += np.log(pi[i])
        result += sub_result@phi[:,i]
    return result


# In[45]:


def M_step(phi, total, X):
    ns = np.sum(phi,axis = 0).reshape(-1,1)
    pi = (ns/total).reshape(-1,1)
    mu = phi.T@X/ns
    covs = []
    for i in range(mu.shape[0]): #for all k
        cov_v = (X- mu[i]).T@np.diag((phi[:,i]))@(X-mu[i])/ns[i]
        covs.append(cov_v)
    return mu, covs, pi


# In[46]:


def find_prob(X, means, covs, pi):
    probas = []
    for i in range(k):
        proba = st.multivariate_normal.pdf(X, mean = means[i], cov = covs[i])
        probas.append(proba)
    probas = np.array(probas).T
    probas = probas@pi
    return probas


# In[133]:


def conf_matrix(X_train_1, X_train_0, k, y_test):
    ans_1 = EM(X_train_1, k)
    ans_0 = EM(X_train_0, k)
    # adding prior : pi(y)
    p1 = find_prob(X_test, ans_1[0], ans_1[1], ans_1[2])*(len(X_train_1)/(len(X_train_1)+len(X_train_0)))
    p0 = find_prob(X_test, ans_0[0], ans_0[1], ans_0[2])*(len(X_train_0)/(len(X_train_1)+len(X_train_0)))
    p = [p0, p1]
    predictor = np.argmax(p,axis = 0)
    result = confusion_matrix(y_test, predictor)
    accuracy = (result[0][0]+result[1][1])/np.sum(result)
    return result, accuracy


# ### Function

# In[48]:


def EM(X, k, iterations=30):
    pi =[1/k]*k
    covs = [np.cov(X.T)]*k
    means = np.random.multivariate_normal(np.mean(X,axis=0),covs[0],size=k)
        
    obj_fcn = []
    for t in range(iterations):
        phi = E_step(means, covs, pi, X)
        obj = loglike(pi, X, means, covs, phi)
        obj_fcn.append(obj)
        means, covs, pi = M_step(phi, X.shape[0], X)
    return means, covs, pi, phi, obj_fcn


# ### A) Implement the EM algorithm for the GMM described in class. Using the training data provided, for each class separately, plot the log marginal objective function for a 3-Gaussian mixture model over 10 different runs and for iterations 5 to 30. There should be two plots, each with 10 curves.

# In[146]:


ans_1 = []
ans_0 = []
for i in range(10):
    ans_1.append(EM(X_train_1, 3, iterations=30)[-1])
    ans_0.append(EM(X_train_0, 3, iterations=30)[-1])


# In[141]:


for i in range(10):
    sns.lineplot(range(5,30), ans_1[i][5:], label = i)


# In[142]:


for i in range(10):
    sns.lineplot(range(0,30), ans_1[i], label = i)


# In[143]:


for i in range(10):
    sns.lineplot(range(5,30), ans_0[i][5:], label = i)


# In[144]:


for i in range(10):
    sns.lineplot(range(0,30), ans_0[i], label = i)


# ### B) Using the best run for each class after 30 iterations, predict the testing data using a Bayes classifier and show the result in a 2 × 2 confusion matrix, along with the accuracy percentage. Repeat this process for a 1-, 2-, 3- and 4-Gaussian mixture model. Show all results nearby each other, and don’t repeat Part (a) for these other cases. Note that a 1-Gaussian GMM doesn’t require an algorithm, although your implementation will likely still work in this case.

# In[49]:


for k in range(1,5):
    print("Result for " + str(k) + "Gaussian Mixture Model:")
    print(conf_matrix(X_train_1, X_train_0, k, y_test))


# In[134]:


for k in range(1,5):
    print("Result for " + str(k) + "Gaussian Mixture Model:")
    print(conf_matrix(X_train_1, X_train_0, k, y_test))


# # Matrix Factorization

# $$
# U_{\mathrm{MAP}}, V_{\mathrm{MAP}}=\arg \max _{U, V} \sum_{(i, j) \in \Omega} \ln p\left(M_{i j} | u_{i}, v_{j}\right)+\sum_{i=1}^{N_{1}} \ln p\left(u_{i}\right)+\sum_{j=1}^{N_{2}} \ln p\left(v_{j}\right)
# $$
# 
# $$
# \mathcal{L}=-\sum_{(i, j) \in \Omega} \frac{1}{2 \sigma^{2}}\left\|M_{i j}-u_{i}^{T} v_{j}\right\|^{2}-\sum_{i=1}^{N_{1}} \frac{\lambda}{2}\left\|u_{i}\right\|^{2}-\sum_{j=1}^{N_{2}} \frac{\lambda}{2}\left\|v_{j}\right\|^{2}+\text { constant }
# $$
# 
# $$
# \begin{aligned} u_{i} &=\left(\lambda \sigma^{2} I+\sum_{j \in \Omega_{u_{i}}} v_{j} v_{j}^{T}\right)^{-1}\left(\sum_{j \in \Omega_{u_{i}}} M_{i j} v_{j}\right) \\ v_{j} &=\left(\lambda \sigma^{2} I+\sum_{i \in \Omega_{y_{j}}} u_{i} u_{i}^{T}\right)^{-1}\left(\sum_{i \in \Omega_{y_{j}}} M_{i j} u_{i}\right) \end{aligned}
# $$

# #### Read data

# In[52]:


ratings = np.genfromtxt('Prob3_ratings.csv', delimiter=',') # user_id, movie_id, rating
ratings_test = np.genfromtxt('Prob3_ratings_test.csv', delimiter=',')

# So index starts from 0 instead of 1 for both user_id and movie_id
ratings[:,0:2] = ratings[:,0:2]-1
ratings_test[:,0:2] = ratings_test[:,0:2]-1

# 100,000 ratings (1-5) from 943 users on 1682 movies. 
n = 943 # number of user
m = 1682 # number of movies

# params
sigma_sq = 0.25

d = 10
lam = 1
times = 10
iterations = 100


# #### Helper functions

# In[53]:


def initialize(ratings, n=943, m = 1682, d = 10, lam = 1):
    mean = np.zeros(d)
    cov = (1/lam)*np.identity(d)
    U = np.random.multivariate_normal(mean, cov, n)
    V = np.random.multivariate_normal(mean, cov, m)
    M = np.ones((n,m))
    for x,y,z in ratings:
        M[int(x)][int(y)] =z
    return U, V, M


# In[54]:


def update_u(V, M, index, lam = 1, sigma_sq = 0.25, d = 10):
    # j_belong is the movies the person i watched.
    j_belong = ratings[ratings[:,0] == index]
    first = np.identity(d)*lam*sigma_sq
    second = np.zeros(d)
    for item in j_belong:
        first += np.outer(V[int(item[1])], V[int(item[1])])
        second += M[int(item[0])][int(item[1])]*V[int(item[1])]
    first = np.linalg.inv(first)
    ans = first@second
    return ans
def update_U(V, M, n=943):
    for i in range(n):
        U[i] = update_u(V, M, i)
    return U


# In[55]:


def update_v(U, M, index, lam = 1, sigma_sq = 0.25, d = 10):
    # i_belong is the people who watched movie j
    i_belong = ratings[ratings[:,1] == index]
    first = np.identity(d)*lam*sigma_sq
    second = np.zeros(d)
    for item in i_belong:
        first += np.outer(U[int(item[0])], U[int(item[0])])
        second += M[int(item[0])][int(item[1])]*U[int(item[0])]
    first = np.linalg.inv(first)
    ans = first@second
    return ans
def update_V(U, M, m=1682):
    for i in range(m):
        V[i] = update_v(U, M, i)
    return V


# In[117]:


def likelihood(U, V, M, ratings, lam = 1):
    lkhd = -lam/2*(np.sum(U**2)+np.sum(V**2))
    lkhd2 = 0 
    phi = U@V.T
    for x,y,z in ratings:
        lkhd2 += (phi[int(x)][int(y)]-z)**2
    lkhd = lkhd - (1/2/sigma_sq)*lkhd2
    return lkhd


# In[61]:


def RMSE(U, V, ratings_test):
    phi = U@V.T
    result = 0
    for x,y,z in ratings_test:
        result += (phi[int(x)][int(y)]-z)**2
    return np.sqrt(result/len(ratings_test))


# ### Function

# In[59]:


def matrix_factorization(ratings, ratings_test, iterations = 100):
    U,V,M = initialize(ratings)    
    likelihoods = []
    for ind in range(iterations):
        U = update_U(V, M)
        V = update_V(U, M)
        lkhd = likelihood(U, V, M, ratings)
        likelihoods.append(lkhd)
    rmse = RMSE(U, V, ratings_test)
    return rmse, likelihoods, U, V


# ### A) Run your code 10 times. For each run, initialize your ui and vj vectors as N (0, I ) random vectors. On a single plot, show the the log joint likelihood for iterations 2 to 100 for each run. In a table, show in each row the final value of the training objective function next to the RMSE on the testing set. Sort these rows according to decreasing value of the objective function.
# 

# In[118]:


RMSEs = []
results = []
U = np.ones((n,d))
V = np.ones((m,d))
best_obj = 0
for i in range(10):
    rmse, lkhds, uu, vv = matrix_factorization(ratings, ratings_test)
    RMSEs.append(rmse)
    results.append(lkhds)
    if lkhds[-1] > best_obj:
        best_obj= lkhds[-1]
        U = uu
        V = vv


# In[119]:


for i in range(10):
    sns.lineplot(range(len(results[i])-2),results[i][2:])


# In[120]:


final_value = []
for i in range(10):
    final_value.append(results[i][-1])
final_value = np.array(final_value)
RMSEs = np.array(RMSEs)
order = np.argsort(final_value)


# In[122]:


for i in range(1,11):
    print("Rank {}".format(i))
    print("Final training objective function value : ", final_value[order[-i]])
    print("RMSE on the testing set : ", RMSEs[order[-i]])
    print("\n")


# ### B) For the run with the highest objective value, pick the movies “Star Wars” “My Fair Lady” and “Goodfellas” and for each movie find the 10 closest movies according to Euclidean distance using their respective locations vj . List the query movie, the ten nearest movies and their distances. A mapping from index to movie is provided with the data.

# In[123]:


# labels
with open('Prob3_movies.txt') as f:
    file=f.read()
file = file.split('\n') 
labels = []
for i in file:
    item = i.split(" (")
    labels.append(item[0])
labels = labels[:-1]
i_star_wars = labels.index("Star Wars")
i_my_fair_lady = labels.index("My Fair Lady")
i_goodfellas = labels.index("GoodFellas")
labels = np.array(labels)


# In[124]:


def find_closest_movies(ind, labels):
    dist = np.sum(abs(V-V[ind]),axis=1)
    closest_movies = np.argsort(dist)[1:11]
    return labels[closest_movies],dist[closest_movies]


# In[125]:


find_closest_movies(i_star_wars, labels)


# In[126]:


find_closest_movies(i_my_fair_lady, labels)


# In[128]:


find_closest_movies(i_goodfellas, labels)


# In[ ]:




