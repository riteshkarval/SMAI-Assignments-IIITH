#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import sys


# In[2]:


colnames = ['class', 'a1', 'a2', 'a3','a4','a5','a6','id']
dataset1 = pd.read_csv("../input_data/RobotDataset/Robot1", names=colnames, header=None,delim_whitespace=True)
dataset2 = pd.read_csv("../input_data/RobotDataset/Robot2", names=colnames, header=None,delim_whitespace=True)
frames = [dataset1,dataset2]
dataset = pd.concat(frames)


# In[3]:


tdataset = dataset.drop('id',axis=1)
tdataset = tdataset.drop('class',axis=1)


# In[4]:


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


# In[5]:


def euclidean(v1,v2):
    ary = spatial.distance.cdist(v1,v2, metric='minkowski')
    return ary[0,0]


# In[6]:


def distances(dataset,sample):
    dist = []
    l = len(dataset)
    for i in range(l):
        dist.append(euclidean(dataset.iloc[[i]],sample))
    return np.asarray(dist)


# In[7]:


def knn(dataset,sample,y,classes,k):
    dist = distances(dataset,sample)
    indices = dist.argsort()[:3]
    counts = np.zeros(len(classes))
    for i in indices:
        counts[classes.index(y.iloc[i])] += 1
    return classes[np.argmax(counts)]


# In[8]:


def train_validate_test_split(dataset):
    size = len(dataset)
    tsize = int(size*0.6)
    vsize = int(size*0.8)
    training_data = dataset.iloc[:tsize].reset_index(drop=True)
    validation_data = dataset.iloc[tsize:vsize].reset_index(drop=True)
    testing_data = dataset.iloc[vsize:].reset_index(drop=True)
    return training_data,validation_data,testing_data


# In[9]:


def knn_algorithm(training_data,test_data,classes,k):
    ttrain = training_data[['a1', 'a2', 'a3','a4','a5','a6']]
    ttest = test_data[['a1', 'a2', 'a3','a4','a5','a6']]
    y = training_data['class']
    pred = []
    for i in range(len(ttest)):
        pred.append(knn(ttrain,ttest.iloc[[i]],y,classes,k))
    return pred


# In[10]:


def confusionmatrix(preds,y,classes):
    n = len(preds)
    noc = len(classes)
    matrix = np.zeros((noc,noc))
    for i in range(n):
        r = classes.index(preds[i])
        c = classes.index(y[i])
        matrix[r][c] += 1
    return matrix


# In[11]:


def stats2(confusionmatrix,classes): 
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


# In[12]:


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
    for i in range(n):
        print("Recall of class",classes[i],":",recall[i])
        print("Precision of class",classes[i],":",precision[i])
        print("F1 Score of class",classes[i],":",f1[i])
        print('\n')
    print("Accuracy:",dval/np.sum(confusionmatrix))
    print("Classification error:",1-(dval/np.sum(confusionmatrix)))
    print("Overall Precision:",np.mean(precision))
    print("Overall Recall:",np.average(recall))
    print("Overall F1 Score:",np.mean(f1))


# In[27]:


def sklearnstats(data):
    datanames = ['training data','validation data','testing data']
    from sklearn.metrics import confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    X = data[0].drop('class',axis=1)
    y = data[0]['class']
    neigh.fit(X, y)
    for i in range(len(data)):
        print("Accuracy on:",datanames[i])
        preds = neigh.predict(data[i].drop('class',axis=1))
        con_mat = confusion_matrix(data[i]['class'], preds)
        stats(con_mat,data[i]['class'].unique())
        print("\n\n\n")


# In[23]:


training_data,validation_data,testing_data = train_validate_test_split(dataset.drop('id',axis=1))
classes = list(training_data['class'].unique())
preds = knn_algorithm(training_data,training_data,classes,3)
cm = confusionmatrix(preds,list(training_data['class']),list(training_data['class'].unique()))
print("Training Data Stats")
stats(cm,training_data['class'].unique())


# In[20]:


preds = knn_algorithm(training_data,validation_data,classes,3)
cm = confusionmatrix(preds,list(validation_data['class']),list(training_data['class'].unique()))
print("Validation Data Stats")
stats(cm,training_data['class'].unique())


# In[29]:


preds = knn_algorithm(training_data,testing_data,classes,3)
cm = confusionmatrix(preds,list(testing_data['class']),list(training_data['class'].unique()))
print("Testing Data Stats")
stats(cm,training_data['class'].unique())


# In[15]:



# SkLearn library results


# In[28]:


sklearnstats([training_data,validation_data,testing_data])

if len(sys.argv) > 1:
    print("Results on input data")
    preds = knn_algorithm(training_data,sys.argv[1],classes,3)
    print(preds)