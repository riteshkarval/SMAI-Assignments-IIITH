#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
from copy import deepcopy


# In[59]:


df = pd.read_csv("reducedData.csv")


# In[60]:


X = df.drop('class',axis=1)
y = df['class']


# In[61]:


from numpy import mean
from numpy import cov
from numpy.linalg import eig


# In[62]:


def euclidean(v1,v2):
    ary = spatial.distance.cdist(v1,v2, metric='minkowski')
    return ary[0,0]


# In[63]:


def distances(a,b):
    return np.linalg.norm(a - b, axis=1)


# In[64]:


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


# In[65]:


def purity(clusters,y,labels):
    purity = 0
    for k in range(5):
        labelcounts = np.zeros(len(labels))
        print(purity)
        for i in range(len(clusters)):
            if clusters[i] == k:
                labelcounts[labels.index(y[i])] = labelcounts[labels.index(y[i])] +1
        print(labelcounts)
        purity += np.max(labelcounts)
    return purity/len(y)


# In[66]:


k = 5
centroids = np.asarray(X[:5])
centroids_old = np.zeros(centroids.shape)


# In[67]:


error = distances(centroids, centroids_old)
error.mean()
X = np.asarray(X)


# In[68]:


clusters = np.zeros(len(X))
for ite in range(10):
    for i in range(len(X)):
        distance = distances(X[i], centroids)
        cluster = np.argmin(distance)
        clusters[i] = cluster
    centroids_old = deepcopy(centroids)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        centroids[i] = np.mean(points, axis=0)
    error = distances(centroids,centroids_old)


# In[69]:


clusters


# In[70]:


y.unique()


# In[71]:


purity(clusters,y,list(y.unique()))


# In[ ]:




