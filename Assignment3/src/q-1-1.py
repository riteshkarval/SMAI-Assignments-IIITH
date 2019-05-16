#!/usr/bin/env python
# coding: utf-8

# In[143]:


import numpy as np
import pandas as pd


# In[144]:


df = pd.read_csv("../input_data/data.csv")


# In[145]:


X = df.drop('xAttack',axis=1)
y = df['xAttack']


# In[146]:


from numpy import mean
from numpy import cov
from numpy.linalg import eig


# In[147]:


X_st = (X-np.mean(X,axis=0))/(np.std(X,axis=0))


# In[148]:


CovarMatrix = np.cov(X_st.T)


# In[149]:


eigen_values, eigen_vectors = np.linalg.eig(CovarMatrix)


# In[150]:


eigen_pair = [(np.abs(eigen_values[i]),eigen_vectors[:,i]) for i in range(len(eigen_vectors))]


# In[151]:


eigen_pair = pd.DataFrame(eigen_pair, columns = ['eigen value','eigen vector'])
eigen_pair.keys()


# In[152]:


eigen_pair = eigen_pair.sort_values(by=['eigen value'], ascending=False)


# In[153]:


threshold=0.7
a = 0
theta = 0
b = np.sum(edf.iloc[:,0],axis = 0)


# In[154]:


for i in range(len(eigen_pair)):
    a += eigen_pair.iloc[i,0]
    ratio = a/b
    var = eigen_pair.iloc[i,0]
    if ratio >= 0.9:
        k = i
        break 


# In[155]:


Vm = eigen_pair.iloc[0:i,1] 
temp = []
for i in Vm:
    temp.append(i)
Vm = np.array(temp)


# In[156]:


ReducedData = X_st.dot(Vm.T)
ReducedData.insert(loc=13, column='class', value=y)
ReducedData.shape


# In[157]:


pd.DataFrame(ReducedData).to_csv("reducedData.csv",index=False)


# In[158]:


ReducedData.head()


# In[ ]:




