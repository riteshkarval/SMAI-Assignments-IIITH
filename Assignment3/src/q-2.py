#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
from scipy import spatial
import seaborn as sns


# In[79]:


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


# In[80]:


def euclidean(v1,v2):
    ary = spatial.distance.cdist(v1,v2, metric='minkowski')
    return ary[0,0]


# In[81]:


def distances(dataset,sample):
    dist = []
    l = len(dataset)
    for i in range(l):
        dist.append(euclidean(dataset.iloc[[i]],sample))
    return np.asarray(dist)


# In[82]:


dataset = pd.read_csv("../input_data/AdmissionDataset/data.csv")


# In[83]:


# Adding bias vector to the dataset with values 1


# In[84]:


dataset.insert(loc=0, column='intercept', value=np.ones(len(dataset)))


# In[85]:


def continious_to_categorical(data):
    median = data.median()
    temp = []
    for v in data:
        if v < median:
            temp.append(0)
        else:
            temp.append(1)
    return pd.Series(temp)


# In[86]:


dataset['Chance of Admit '] = continious_to_categorical(dataset['Chance of Admit '])


# In[87]:


dataset.head()


# In[88]:


def train_validate_test_split(dataset):
    size = len(dataset)
    tsize = int(size*0.6)
    vsize = int(size*0.8)
    training_data = dataset.iloc[:tsize].reset_index(drop=True)
    validation_data = dataset.iloc[tsize:vsize].reset_index(drop=True)
    testing_data = dataset.iloc[vsize:].reset_index(drop=True)
    return training_data,validation_data,testing_data


# In[89]:


def predict(X,weights):
    values = np.dot(X,weights)
    for i in range(len(values)):
        if (1/(1+np.exp(-1 * values[i]))) > 0.5:
            values[i] = 1
        else:
            values[i] = 0
    return values


# In[90]:


def meansquareerror(X,weights,y):
    preds = predict(X,weights)
    return ((preds - y)**2).mean()


# In[91]:


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


# In[92]:


def confusionmatrix(preds,y,classes):
    n = len(preds)
    noc = len(classes)
    matrix = np.zeros((noc,noc))
    for i in range(n):
        r = classes.index(preds[i])
        c = classes.index(y[i])
        matrix[r][c] += 1
    return matrix


# In[93]:


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


# In[94]:


training_data,validation_data,testing_data = train_validate_test_split(dataset)
X = training_data.drop('Chance of Admit ',axis=1)
y = training_data['Chance of Admit ']
# for costf in costfunctions:
weights= logisticRegressionGD(X,y)
print("Training Data Accuracy:")
preds = predict(training_data.drop('Chance of Admit ',axis = 1),weights)
cm = confusionmatrix(preds,training_data['Chance of Admit '],[0,1])
print(stats(cm,[0,1]))
print("\nValidation Data Accuracy")
preds = predict(validation_data.drop('Chance of Admit ',axis = 1),weights)
cm = confusionmatrix(preds,validation_data['Chance of Admit '],[0,1])
print(stats(cm,[0,1]))
print("\nTesting Data Accuracy")
preds = predict(testing_data.drop('Chance of Admit ',axis = 1),weights)
cm = confusionmatrix(preds,testing_data['Chance of Admit '],[0,1])
print(stats(cm,[0,1]))


# In[95]:


def knn_algorithm(training_data,test_data,classes,k):
    ttrain = training_data[['intercept', 'Serial No.', 'GRE Score', 'TOEFL Score',
       'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
    ttest = test_data[['intercept', 'Serial No.', 'GRE Score', 'TOEFL Score',
       'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
    y = training_data['Chance of Admit ']
    pred = []
    for i in range(len(ttest)):
        pred.append(knn(ttrain,ttest.iloc[[i]],y,classes,k))
    return pred


# In[96]:


def knn(dataset,sample,y,classes,k):
    dist = distances(dataset,sample)
    indices = dist.argsort()[:3]
    counts = np.zeros(len(classes))
    for i in indices:
        counts[classes.index(y.iloc[i])] += 1
    return classes[np.argmax(counts)]


# In[97]:


preds = knn_algorithm(training_data,training_data,[0,1],3)
cm = confusionmatrix(preds,list(training_data['Chance of Admit ']),list(training_data['Chance of Admit '].unique()))
print("Training Data Stats")
stats(cm,training_data['Chance of Admit '].unique())


# In[98]:


preds = knn_algorithm(training_data,testing_data,[0,1],3)
cm = confusionmatrix(preds,list(testing_data['Chance of Admit ']),list(training_data['Chance of Admit '].unique()))
print("Testing Data Stats")
stats(cm,training_data['Chance of Admit '].unique())


# In[99]:


preds = knn_algorithm(training_data,validation_data,[0,1],3)
cm = confusionmatrix(preds,list(validation_data['Chance of Admit ']),list(training_data['Chance of Admit '].unique()))
print("Validation Data Stats")
stats(cm,training_data['Chance of Admit '].unique())


# In[ ]:




