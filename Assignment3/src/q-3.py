#!/usr/bin/env python
# coding: utf-8

# In[214]:


import pandas as pd
import numpy as np
from scipy import spatial
import seaborn as sns


# In[215]:


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


# In[216]:


dataset = pd.read_csv("../input_data/wine-quality/data.csv",sep=';')


# In[217]:


# Adding bias vector to the dataset with values 1


# In[218]:


dataset.insert(loc=0, column='intercept', value=np.ones(len(dataset)))


# In[219]:


def continious_to_categorical(data):
    median = data.median()
    low = data.min()
    high = data.max()
    lmedian = (low+median)/2
    rmedian = (high+median)/2
    temp = []
    for v in data:
        if v < (lmedian+median)/2:
            temp.append('low')
        elif v > (rmedian+median)/2:
            temp.append('high')
        else:
            temp.append('medium')
    return pd.Series(temp)


# In[220]:


dataset['alcohol'] = continious_to_categorical(dataset['alcohol'])


# In[221]:


dataset['alcohol'].unique()


# In[222]:


def train_validate_test_split(dataset):
    size = len(dataset)
    tsize = int(size*0.6)
    vsize = int(size*0.8)
    training_data = dataset.iloc[:tsize].reset_index(drop=True)
    validation_data = dataset.iloc[tsize:vsize].reset_index(drop=True)
    testing_data = dataset.iloc[vsize:].reset_index(drop=True)
    return training_data,validation_data,testing_data


# In[223]:


def predict(X,weights):
    values = np.dot(X,weights)
    for i in range(len(values)):
        if (1/(1+np.exp(-1 * values[i]))) > 0.5:
            values[i] = 1
        else:
            values[i] = 0
    return values


# In[224]:


def predictonevsall(classifiers,X):
    classes = list(classifiers.keys())
    preds = []
    for i in range(len(X)):
        pred = np.zeros(len(classes))
        j = 0
        for cls in classes:
            pred[j] = (1/(1+np.exp(-1 * np.dot(X.iloc[[i]],classifiers[cls])))) 
            j += 1
        preds.append(classes[np.argmax(pred)])
    return preds


# In[225]:


def logisticRegressionGD(X,y,lr=0.001,iter=60):
    weights = np.random.rand(len(X.columns))
    for i in range(iter):
        preds = predict(X,weights)
        weights[0] = weights[0] - lr *(preds - y).mean()
        for j in range(1,len(X.columns)):
            temp = X.iloc[:, [j]].values
            x = np.reshape(temp,temp.shape[0])
            weights[j] = weights[j] - (lr *((preds - y)* x).mean())
#         print(cost)
    return weights


# In[233]:


def accuracy(preds,Y):
    count = 0
    for i in range(len(Y)):
        if preds[i] == Y[i]:
            count += 1
    return count/len(Y)


# In[227]:


def stats(confusionmatrix,classes): 
    n = len(classes)
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    colsums = confusionmatrix.sum(axis=0)
    rowsums = confusionmatrix.sum(axis=1)
    dval = 0
    for i in range(n):
        precision[i] = confusionmatrix[i,i]/colsums[i]
        recall[i] = confusionmatrix[i,i]/rowsums[i]
        f1[i] = safe_div(2,(safe_div(1,precision[i]))+safe_div(1,recall[i]))
        dval += confusionmatrix[i,i]
    return dval/np.sum(confusionmatrix)


# In[230]:


training_data,validation_data,testing_data = train_validate_test_split(dataset)
classes = dataset['alcohol'].unique()
print(classes)
classifiers = {}
for cls in classes:
    temp = training_data.copy()
    for i in range(len(temp['alcohol'])):
        if temp['alcohol'][i] == cls:
            temp['alcohol'] = 1
        else:
            temp['alcohol'] = 0
    X = temp.drop('alcohol',axis = 1)
    y = temp['alcohol']
    classifiers[cls] = logisticRegressionGD(X,y)  


# In[231]:


preds = predictonevsall(classifiers,training_data.drop('alcohol',axis = 1))


# In[234]:


accuracy(preds,training_data['alcohol'])


# In[ ]:




