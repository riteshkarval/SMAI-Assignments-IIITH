#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn import metrics


# In[23]:


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


# In[24]:


df = pd.read_csv("reducedData.csv")


# In[25]:


X = df.drop('class',axis=1)
y = df['class']


# In[26]:


gmm = GaussianMixture(n_components=5)
gmm.fit(X)


# In[27]:


clusters = gmm.predict(X)


# In[28]:


purity(clusters,y,list(y.unique()))


# In[ ]:




