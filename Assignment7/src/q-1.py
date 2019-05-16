#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
from scipy import spatial
import seaborn as sns
import matplotlib.pyplot as plt
import sys


# In[2]:


def normalizeColumn(column):
    mean = column.mean()
    std = column.std()
    return (column - mean)/ std


# In[3]:


dataset = pd.read_csv("../input_data/AdmissionDataset/data.csv")


# In[4]:


dataset = dataset.drop('Serial No.', axis = 1)
dataset['GRE Score'] = normalizeColumn(dataset['GRE Score'])
dataset['TOEFL Score'] = normalizeColumn(dataset['TOEFL Score'])
dataset['University Rating'] = normalizeColumn(dataset['University Rating'])
dataset['LOR '] = normalizeColumn(dataset['LOR '])
dataset['CGPA'] = normalizeColumn(dataset['CGPA'])


# In[5]:


# Adding bias vector to the dataset with values 1


# In[6]:


dataset.insert(loc=0, column='intercept', value=np.ones(len(dataset)))


# In[7]:


dataset.head()


# In[8]:


dataset.keys()


# In[9]:


# Plotting the dataset attributes with respect to output 


# In[ ]:





# In[10]:


def train_validate_test_split(dataset):
    size = len(dataset)
    tsize = int(size*0.6)
    vsize = int(size*0.8)
    training_data = dataset.iloc[:tsize].reset_index(drop=True)
    validation_data = dataset.iloc[tsize:vsize].reset_index(drop=True)
    testing_data = dataset.iloc[vsize:].reset_index(drop=True)
    return training_data,validation_data,testing_data


# In[11]:


def predict(X,weights):
    return np.dot(X,weights)


# In[12]:


def meansquareerror(preds,y):
    return ((preds - y)**2).mean()


# In[13]:


def lassolinearRegressionGD(X,y,weights,lr=0.001,lambdavalue = 0,iter=1):
    for i in range(iter):
        preds = predict(X,weights)
        error = meansquareerror(preds,y) + lambdavalue * (np.sum(np.absolute(weights[1:])))
        weights[0] = weights[0] + lr *error
        for j in range(1,len(X.columns)):
            weights[j] = weights[j] + (lr *error* np.mean(X.iloc[:, [j]].values))
    return weights


# In[14]:


def statistics(weights,data):
    X = data.drop('Chance of Admit ',axis=1)
    y = data['Chance of Admit ']
    preds = predict(X,weights)
    cost = meansquareerror(preds,y)
    return cost


# In[15]:


training_data,validation_data,testing_data = train_validate_test_split(dataset)
X_train = training_data.drop('Chance of Admit ',axis=1)
y_train = training_data['Chance of Admit ']


# In[35]:


weights = np.zeros((len(X_train.columns)))
errors = {'lambda_value':[],
          'train_error':[],
           'validation_error':[],
            'test_error':[]}
for i in range(10):
    errors['lambda_value'].append(i)
    weights= lassolinearRegressionGD(X_train,y_train,weights,lambdavalue=i,iter = 10)
    errors['train_error'].append(statistics(weights,training_data))
    errors['validation_error'].append(statistics(weights,validation_data))
    errors['test_error'].append(statistics(weights,testing_data))


# In[36]:


df = pd.DataFrame(errors)


# In[37]:


ax = plt.gca()
df.plot(kind='line',x='lambda_value',y='train_error',color = 'green',ax=ax)
df.plot(kind='line',x='lambda_value',y='validation_error', color='red', ax=ax)
df.plot(kind='line',x='lambda_value',y='test_error', color='blue', ax=ax)
plt.show()


# In[27]:


sns.pairplot(pd.DataFrame(errors), x_vars='lambda_value', y_vars=['train_error','validation_error','test_error',], height=5, aspect=2)


# In[ ]:

if len(sys.argv) > 1:
    inp = sys.argv[1]
    predict(X)


