#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import pickle


# In[42]:


df = pd.read_csv("../Apparel/apparel-test.csv")


# In[43]:


X = np.asarray(df)/255


# In[44]:


X.shape


# In[45]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# In[46]:


def sigmoid_derivative(x):
    return x * (1 - x)


# In[47]:


def relu(x):
    return np.where(x > 0, 1.0, 0.0)


# In[48]:


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)


# In[49]:


def tanh(x):
    return (2/(1+np.exp(-2*x))) - 1


# In[50]:


def tanh_derivative(x):
    return 1 - (x * x) 


# In[67]:


class NN:
    def __init__(self,learningrate = 0.01):
        self.shape = []
        self.activation = ''
        self.learningrate = learningrate
        self.activations = {'sigmoid':[sigmoid,sigmoid_derivative],
                           'relu':[relu,relu_derivative],
                           'tanh':[tanh,tanh_derivative]}
        self.layers = []
        self.weights = []

    def forwardpass(self,data):
        self.layers[0][0:-1] = data
        for i in range(1,len(self.shape)):
            self.layers[i][...] = self.activations[self.activation][0](np.dot(self.layers[i-1],self.weights[i-1]))
        return self.layers[-1]


# In[77]:


def prediction(network,samples):
    pred = []
    n = samples.shape[0]
    for i in range(n):
        out = network.forwardpass(samples[i])
        pred.append([np.argmax(out)])
    return pred


# In[78]:


with open(r"network.pickle", "rb") as input_file:
    e = pickle.load(input_file)


# In[79]:


nn = NN()


# In[80]:


nn.shape = e['layers']
nn.weights = e['weights']
nn.activation = e['activation']
nn.layers = e['layerbias']


# In[81]:


len(e['weights'])


# In[82]:


preds = prediction(nn,X)


# In[83]:


df = pd.DataFrame.from_records(preds, columns=['label'])

# In[84]:


df.to_csv("../output_data/2018900060_prediction.csv", sep=',',index=False)

