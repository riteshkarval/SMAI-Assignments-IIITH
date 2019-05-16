#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


# In[19]:


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


# In[20]:


df = pd.read_csv("reducedData.csv")


# In[21]:


X = df.drop('class',axis=1)
y = df['class']
len(X)


# In[22]:


cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')  
clusters = cluster.fit_predict(X[:20000])  


# In[23]:


purity(clusters,y[:20000],list(y.unique()))


# In[ ]:




