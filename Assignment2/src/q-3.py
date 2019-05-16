#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
from scipy import spatial
import seaborn as sns
import sys

# In[68]:


dataset = pd.read_csv("../input_data/AdmissionDataset/data.csv")


# In[ ]:


# Adding bias vector to the dataset with values 1


# In[69]:


dataset.insert(loc=0, column='intercept', value=np.ones(len(dataset)))


# In[82]:


# Plotting the dataset attributes with respect to output 


# In[70]:


sns.pairplot(dataset, x_vars=['Serial No.', 'GRE Score', 'TOEFL Score',
       'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research',], y_vars='Chance of Admit ', height=5, aspect=0.7)


# In[71]:


def train_validate_test_split(dataset):
    size = len(dataset)
    tsize = int(size*0.6)
    vsize = int(size*0.8)
    training_data = dataset.iloc[:tsize].reset_index(drop=True)
    validation_data = dataset.iloc[tsize:vsize].reset_index(drop=True)
    testing_data = dataset.iloc[vsize:].reset_index(drop=True)
    return training_data,validation_data,testing_data


# In[72]:


def predict(X,weights):
    return np.dot(X,weights)


# In[73]:


def meansquareerror(X,weights,y):
    preds = predict(X,weights)
    return ((preds - y)**2).mean()


# In[74]:


def meanabsoluteerror(X,weights,y):
    preds = predict(X,weights)
    return (np.abs(preds - y)).mean()


# In[75]:


def meanpercentageerror(X,weights,y):
    preds = predict(X,weights)
    return 100*((preds - y)**2).mean()


# In[76]:


def linearRegressionGD(X,y,lr=0.001,iter=60):
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


# In[79]:


def statistics(weights,data):
    X = data.drop('Chance of Admit ',axis=1)
    y = data['Chance of Admit ']
    costfunctionnames = ['mean square error','mean absolute error','mean percentage error']
    costfunctions = [meansquareerror,meanabsoluteerror,meanpercentageerror]
    for i in range(len(costfunctions)):
        cost = costfunctions[i](X,weights,y)
        print(costfunctionnames[i],":",cost)


# In[89]:


def sklearnlinearreg(data):
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    regr = linear_model.LinearRegression()
    X = data[0].drop('Chance of Admit ',axis=1)
    y = data[0]['Chance of Admit ']
    regr.fit(X,y)
    print("Stats Training Data:")
    statistics(regr.coef_,data[0])
    print("\n\n")
    print("Stats on validation data")
    statistics(regr.coef_,data[1])
    print("\n\n")
    print("Stats on testing data")
    statistics(regr.coef_,data[2])


# In[90]:


training_data,validation_data,testing_data = train_validate_test_split(dataset)
X = training_data.drop('Chance of Admit ',axis=1)
y = training_data['Chance of Admit ']
# for costf in costfunctions:
weights= linearRegressionGD(X,y)
print("Training Data:")
statistics(weights,training_data)
print("\nValidation Data")
statistics(weights,validation_data)
print("\nTesting Data")
statistics(weights,testing_data)


# In[91]:


print(weights)


# In[93]:


# SkLearn library results


# In[92]:


sklearnlinearreg([training_data,validation_data,testing_data])


# In[ ]:
if len(sys.argv) > 1:
    print("Results on input data")
    preds = predict(sys.argv[1],weights)
    print(preds)



