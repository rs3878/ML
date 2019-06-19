#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# # Markov Chain
# In this problem, you will rank 767 college football teams based on the scores of every game in the 2018 season. The data provided in CFB2018 scores.csv contains the result of one game on each line in the format Team A index, Team A points, Team B index, Team B points. If Team A has more points than Team B, then Team A wins, and vice versa. The index of a team refers to the row of “TeamNames.txt” where that team’s name can be found.

# Construct a 767×767 random walk matrix M on the college football teams. First construct the unnor- malized matrix $\widehat{M}$ with values initialized to zeros. For one particular game, let i be the index of Team A and j the index of Team B. Then update
# 
# $$
# \begin{aligned} \widehat{M}_{i i} & \leftarrow \widehat{M}_{i i}+1\{\text { Team } \mathrm{A} \text { wins }\}+\frac{\text { points }_{i}}{\text { points }_{i}+\text { points }_{j}} \\ \widehat{M}_{j j} & \leftarrow \widehat{M}_{j j}+1\{\text { Team } \mathrm{B} \text { wins }\}+\frac{\text { points }_{j}}{\text { points }_{i}+\text { points }_{j}} \end{aligned}
# $$
# $$
# \widehat{M}_{i j} \leftarrow \widehat{M}_{i j}+1\{\text { Team } \mathrm{B} \text { wins }\}+\frac{\text { points }_{j}}{\text { points }_{i}+\text { points } s_{j}}
# $$
# $$
# \widehat{M}_{j i} \leftarrow \widehat{M}_{j i}+1\{\text { Team } \mathrm{A} \text { wins }\}+\frac{\text { points }_{i}}{\text { points }_{i}+\text { points }_{j}}
# $$
# After processing all games, let M be the matrix formed by normalizing the rows of $\widehat{M}$ so they sum to one.

# In[259]:


scores = pd.read_csv("CFB2018_scores.csv", header = None).values

# reindex
scores[:,0] = scores[:,0]-1 
scores[:,2] = scores[:,2]-1

# get team names
names = []
with open('TeamNames.txt') as f:
    file=f.read()
file = file.split('\n')
for i in file:
    names.append(i)
names = np.array(names[:-1])

n = 767 # 767 teams total
M = np.zeros((n,n))
    
# construct M Matrix using scores
for A_i,A_pt,B_i,B_pt in scores:
    if A_pt == B_pt:
        print("Oops there is a problem") # check for tie
    total = A_pt + B_pt
    M[A_i][A_i] += (1*(A_pt>B_pt) + A_pt/total)
    M[B_i][B_i] += (1*(B_pt>A_pt) + B_pt/total)
    M[A_i][B_i] += (1*(B_pt>A_pt) + B_pt/total)
    M[B_i][A_i] += (1*(A_pt>B_pt) + A_pt/total)
    
M = M/np.sum(M, axis = 1, keepdims = True) # normalizing


# Let $w_t$ be the 1×767 state vector at step t. Set $w_0$ to the uniform distribution. Therefore, $w_t$ is the marginal distribution on each state after t steps given that the starting state is chosen uniformly at random.
# $$
# w_{t+1}(j)=\sum_{u=1}^{S} M_{u j} w_{t}(u) \quad \Longleftrightarrow \quad w_{t+1}=w_{t} M
# $$

# ### a) Use wt to rank the teams by sorting in decreasing value according to this vector. List the top 25 team names (see accompanying file) and their corresponding values in wt for t = 10, 100, 1000, 10000.
# 

# In[260]:


w = np.random.uniform(low=0, high = 1, size = n) #starting state is chose uniformly at random
w = w/np.sum(w)
#w = [1/n]*n #uniform distribution starting state
for t in (10,100,1000,10000):
    w_ = w@np.linalg.matrix_power(M,t)
    values_ = np.sort(w_)[::-1][:25]
    indexes_ = np.argsort(w_)[::-1][:25]
    names_ = names[indexes_]
    print("\n\nAt t = {}\n".format(t),values_,"\n",names_)


# ### b) We saw that $w_∞$ is related to the first eigenvector of $M^T$ . That is, we can find $w_∞$ by getting the first eigenvector and eigenvalue of $M^T$ and post-processing:
# $$
# M^{T} u_{1}=\lambda_{1} u_{1}, \quad w_{\infty}=u_{1}^{T} /\left[\sum_{j} u_{1}(j)\right]
# $$
# $$
# \begin{array}{l}{\text { This is because } u_{1}^{T} u_{1}=1 \text { by convention. Also, we observe that } \lambda_{1}=1 \text { for this specific matrix. }} \\ {\text { Plot }\left\|w_{t}-w_{\infty}\right\|_{1} \text { as a function of } t \text { for } t=1, \ldots, 10000 .}\end{array}
# $$

# In[243]:


eigenvalue, eigenvector = np.linalg.eig(M.T)
max_index = np.argmax(eigenvalue)
u = eigenvector[:,max_index]
w_inf = u/(np.sum(u))
np.sort(w_inf)[::-1][:25], names[np.argsort(w_inf)[::-1][:25]]


# In[255]:


w = np.random.uniform(low=0, high = 1, size = n) #starting state is chose uniformly at random
w = w/np.sum(w)
wt = w
diffs = []
for i in range(1,10000):
    wt = wt@M
    diff = np.sum(np.abs(wt-w_inf))
    #diff = np.linalg.norm((wt-w_inf),ord =1)
    diffs.append(diff)
ax = sns.lineplot(range(9999),diffs)
ax.set_xlabel("iterations")
ax.set_title("Difference b/w w_t and w_inf")


# # Nonnegative matrix factorization
# In this problem you will factorize an N × M matrix X into a rank-K approximation W H , where W is N × K, H is K × M and all values in the matrices are nonnegative. Each value in W and H can be initialized randomly to a positive number, e.g., from a Uniform(1,2) distribution.
# The data to be used for this problem consists of 8447 documents from The New York Times. (See below for how to process the data.) The vocabulary size is 3012 words. You will need to use this data to construct the matrix X, where $X_{ij}$ is the number of times word i appears in document j. Therefore, X is 3012×8447 and most values in X will equal zero.

# In[2]:


N = 3012 # The vocabulary size is 3012 words.
M = 8447 # 8447 documents from The New York Times.

K = 25 #  Rank = 25, meaning 25 topics
W = np.random.uniform(low=1.0, high=2.0, size=(N,K))
H = np.random.uniform(low=1.0, high=2.0, size=(K,M))

# read in X
X = np.zeros((N,M)) # X_ij is the number of times word i appears in document j. 
with open('nyt_data.txt') as f:
    file = f.read()
file = file.split('\n')
file = file[:-1]
index = 0
for item in file:
    items = item.split(",")
    for i in items:
        A,B = i.split(":")
        X[int(A)-1][index] = int(B)
    index +=1


# ### a) Implement and run the NMF algorithm on this data using the divergence penalty. Set the rank to 25 and run for 100 iterations. This corresponds to learning 25 topics. Plot the objective as a function of iteration.
# 

# #### Divergence Penalty Objective Function
# $$
# \begin{array}{l} \\ {\quad \min \sum_{i j}\left[X_{i j} \ln \frac{1}{(w H)_{i j}}+(W H)_{i j}\right] \text { subject to } W_{i k} \geq 0, H_{k j} \geq 0}\end{array}
# $$
# $$
# \begin{array}{l}{\text { Randomly initialize } H \text { and } W \text { with nonnegative values. }} \\ {\text { Iterate the following, first for all values in } H, \text { then all in } W :}\end{array}
# $$
# $$
# \begin{aligned} H_{k j} & \leftarrow H_{k j} \sum_{i} \frac{\sum_{i} W_{i k} X_{i j} /(W H)_{i j}}{\sum_{i} W_{i k}} \\ W_{i k} & \leftarrow W_{i k} \frac{\sum_{j} H_{k j} X_{i j} /(W H)_{i j}}{\sum_{j} H_{k j}} \end{aligned}
# $$
# $$
# \text { until the change in } D(X \| W H) \text { is "small." }
# $$

# In[3]:


iterations = 100
divergence_obj = np.zeros(iterations)

#reinitialize
W = np.random.uniform(low=1.0, high=2.0, size=(N,K))
H = np.random.uniform(low=1.0, high=2.0, size=(K,M))

for i in range(iterations):
    helper_purple = X/((W@H)+1e-16)
    helper_pink = W.T/np.sum(W.T, axis = 1, keepdims = True)
    H = H*(helper_pink@helper_purple) #update H
    helper_purple = X/((W@H)+1e-16)
    helper_green = H.T/np.sum(H.T, axis = 0, keepdims = True)
    W = W*(helper_purple@helper_green) #update W
    divergence_obj[i] = -np.sum(X*np.log(W@H+1e-16)-W@H)


# In[4]:


sns.lineplot(range(100),divergence_obj)


# ### b) After running the algorithm, normalize the columns of W so they sum to one. For each column of W , list the 10 words having the largest weight and show the weight. The ith row of W corresponds to the ith word in the “dictionary” provided with the data. Organize these lists in a 5 × 5 table.

# In[5]:


vocab = pd.read_csv('nyt_vocab.dat', sep='\s+', header = None).values
W_final = W/np.sum(W, axis = 0, keepdims = True)


# In[16]:


result = []
for i in range(K):
    index_ = np.argsort(W_final[:,i])[::-1][:10]
    value_ = np.sort(W_final[:,i])[::-1][:10]
    word_ = vocab[index_]
    result.append(word_)


# In[18]:


result_with_weight = result.copy()
for i in range(len(result_with_weight)):
    value_ = np.sort(W_final[:,i])[::-1][:10]
    for j in range(len(result_with_weight[i])):
        result_with_weight[i][j] = result_with_weight[i][j]+" " +str(value_[j])
result_with_weight


# In[19]:


pd.DataFrame(result_with_weight).to_csv("result.csv")


# In[507]:


result


# In[ ]:




