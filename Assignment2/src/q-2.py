#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy import spatial
import sys

# In[3]:


colnames = ['id','age','yoe','income','zip','family','monthlyexp','education',
           'mortagev','class','otherbankacc','certificate','internetbanking','creditcard']
dataset = pd.read_csv("../input_data/LoanDataset/data.csv", names=colnames, header=None)
dataset = dataset.drop([0])


# In[4]:


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


# In[5]:


def train_validate_test_split(dataset):
    size = len(dataset)
    tsize = int(size*0.6)
    vsize = int(size*0.8)
    training_data = dataset.iloc[:tsize].reset_index(drop=True)
    validation_data = dataset.iloc[tsize:vsize].reset_index(drop=True)
    testing_data = dataset.iloc[vsize:].reset_index(drop=True)
    return training_data,validation_data,testing_data


# In[6]:


def meanstdv(col):
    mean = col.mean()
    stdv = col.std()
    return mean,stdv


# In[7]:


def pdf(x,mean,stdv):
    exp = np.exp(-(np.power(x-mean,2)/(2*np.power(stdv,2))))
    return (1/(np.sqrt(2*np.pi)*stdv))*exp


# In[8]:


def classprobablities(column):
    counts = column.value_counts()
    prob = np.zeros(len(column.unique()))
    for i in range(len(prob)):
        prob[i] = counts.iloc[[i]].iloc[0]/column.size
    return prob    


# In[9]:


def summaries(dataset):
    summary = {}
    attributes = dataset.keys()
    for att in attributes:
        summary[att] = []
        summary[att].append([meanstdv(dataset[att])])
#     print(summary)
    return summary    


# In[10]:


def conditionalprobablities(dataset):
    sets = []
    classes = dataset['class'].unique()
    for c in classes:
        sets.append([(dataset.loc[dataset['class'] == c]).drop('class',axis=1)])
    summary = []
    for s in sets:
        summary.append(summaries(s[0]))
    return summary


# In[11]:


def predict(classprob,summary,classes,sample):
    l = sample.size
    pred = []
    noc = len(classes)
#     print(l,noc,classes[0])
    attr = sample.keys()
    for i in range(noc):
        csummary = summary[i]
        cprob = 1
        for j in range(l):
            tmean, tstdv = csummary[attr[j]][0][0][0],csummary[attr[j]][0][0][0]
            cprob *= pdf(sample.iloc[j],tmean,tstdv)
#             print(cprob)
        pred.append(cprob*classprob[i])
    pred = np.asarray(pred)
    return classes[np.argmax(pred)]


# In[12]:


def getpredictions(classprob,summary,classes,data):
    predictions = []
    l = len(data)
    for i in range(l):
        predictions.append(predict(classprob,summary,classes,data.iloc[i]))
    return predictions


# In[22]:


def sklearnnaivebayes(data):
    from sklearn import metrics
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    X = data[0].drop('class',axis=1)
    y = data[0]['class']
    model.fit(X,y)
    print("Stats on training data")
    predictions = model.predict(X)
    stats(predictions,data[0]['class'])
    print("\n\n")
    print("Stats on validation data")
    predictions = model.predict(data[1].drop('class',axis=1))
    stats(predictions,data[1]['class'])
    print("\n\n")
    print("Stats on testing data")
    predictions = model.predict(data[2].drop('class',axis=1))
    stats(predictions,data[2]['class'])


# In[15]:


training_data,validation_data,testing_data = train_validate_test_split(dataset)
summary = conditionalprobablities(training_data)
classprob = classprobablities(training_data['class'])
print("Stats on training data")
predictions = getpredictions(classprob,summary,training_data['class'].unique(),training_data.drop('class',axis=1))
stats(predictions,training_data['class'])

# for i in range(5):
#     print('i am i',i)
#     print(predict(classprob,summary,dataset['class'].unique(),tdataset.iloc[i]))


# In[16]:


print("Stats on validation data")
predictions = getpredictions(classprob,summary,training_data['class'].unique(),validation_data.drop('class',axis=1))
stats(predictions,validation_data['class'])


# In[17]:


print("Stats on testing data")
predictions = getpredictions(classprob,summary,training_data['class'].unique(),testing_data.drop('class',axis=1))
stats(predictions,testing_data['class'])


# In[ ]:


# SkLearn library results


# In[23]:


sklearnnaivebayes([training_data,validation_data,testing_data])


# In[ ]:
if len(sys.argv) > 1:
    print("Results on input data")
    preds = getpredictions(classprob,summary,training_data['class'].unique(),sys.argv[1])
    print(preds)



