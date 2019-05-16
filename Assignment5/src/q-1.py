#!/usr/bin/env python
# coding: utf-8

# In[354]:


import pandas as pd
import numpy as np
np.random.seed(1)


# In[355]:


df = pd.read_csv("../Apparel/apparel-trainval.csv")


# In[356]:


X = np.asarray(df.drop('label', axis = 1))/255
Y = np.asarray(df['label'])
Y = np.reshape(Y,(Y.shape[0],1))


# In[357]:


temp = np.zeros((Y.shape[0],10))
for i in range(Y.shape[0]):
    temp[i,Y[i]] = 1


# In[358]:


Y = temp


# In[359]:


def traintestvalidatesplit(data):
    x,y = data[0], data[1]
    n = x.shape[0]
    k = int(n*0.8)
    x_train = x[:k]
    y_train = y[:k]
    x_val = x[k:]
    y_val = y[k:]
    return [x_train,y_train],[x_val,y_val]


# In[360]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# In[361]:


def sigmoid_derivative(x):
    return x * (1 - x)


# In[362]:


def relu(x):
    return np.where(x > 0, 1.0, 0.0)


# In[363]:


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)


# In[364]:


def tanh(x):
    return (2/(1+np.exp(-2*x))) - 1


# In[365]:


def tanh_derivative(x):
    return 1 - (x * x) 


# In[412]:


def prediction(network,samples):
    pred = []
    n = samples.shape[0]
    for i in range(n):
        out = network.forwardpass(samples[i])
        pred.append([np.argmax(out)])
    return pred


# In[366]:


class NN:
    def __init__(self,layerSizes,activation,learningrate = 0.01):
        self.shape = layerSizes
        self.activation = activation
        self.learningrate = learningrate
        self.activations = {'sigmoid':[sigmoid,sigmoid_derivative],
                           'relu':[relu,relu_derivative],
                           'tanh':[tanh,tanh_derivative]}
        n = len(layerSizes)
        self.layers = []
        self.layers.append(np.ones(self.shape[0]+1))
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))
        self.weights = []
        for i in range(n-1):
            temp = np.zeros((self.layers[i].size,self.layers[i+1].size), dtype = 'd')
            self.weights.append(np.random.randn(*temp.shape))
        self.derivative = [0,]*len(self.weights)

    def forwardpass(self,data):
        self.layers[0][0:-1] = data
        for i in range(1,len(self.shape)):
            self.layers[i][...] = self.activations[self.activation][0](np.dot(self.layers[i-1],self.weights[i-1]))
        return self.layers[-1]


    def backpropogation(self, target, momentum=0.1):
        error = target - self.layers[-1]
        weight_deltas = []
        weight_delta = error*self.activations[self.activation][1](self.layers[-1])
        weight_deltas.append(weight_delta)

        for i in range(len(self.shape)-2,0,-1):
            weight_delta = np.dot(weight_deltas[0],self.weights[i].T)*self.activations[self.activation][1](self.layers[i])
            weight_deltas.insert(0,weight_delta)
            
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            weight_delta = np.atleast_2d(weight_deltas[i])
            der = np.dot(layer.T,weight_delta)
            self.weights[i] += self.learningrate*der + momentum*self.derivative[i]
            self.derivative[i] = der

        return (error**2).mean()


# In[367]:


def trainMLP(network,samples, epochs=10, momentum=0.1):
    error_set = []
    for i in range(epochs):
        print('Epoch: ', i+1)
        n = samples[0].shape[0]
        error = 0
        for j in range(n):
            out = network.forwardpass(samples[0][j] )
            error += network.backpropogation( samples[1][j], momentum )
        error_set.append(error/n)
        print('Training error',error/n)
    return error_set, error/n


# In[368]:


def validateMLP(network,samples):
    n = samples[0].shape[0]
    error = 0
    for j in range(n):
        out = network.forwardpass(samples[0][j] )
        error += network.backpropogation(samples[1][j])
    print('Validation error',error/n)


# In[369]:


def testMLP(network,samples):
    error = 0
    n = samples[0].shape[0]
    for i in range(n):
        out = network.forwardpass(samples[0][i])
        error += ((samples[1] - out)**2).mean()
    print('Testing Error: ',error/n)


# In[370]:


trainingdata, validationdata = traintestvalidatesplit([X,Y])
sample = [X[:10],Y[:10]]


# In[371]:


nn = NN([X.shape[1],256,64,10],'sigmoid',0.01)
epochs = 17
error_set, finalerror = trainMLP(nn,trainingdata,epochs)


# In[373]:


validateMLP(nn,validationdata)


# In[376]:


# As a effect of activation functions it has been found that ReLu saturates after few epochs(usually after 2 or 3)
# , whereas TanH found to jumping nerby local minima, which relsulted in jumping inbetween a range of error. 

# However, in case of MLP sigmoid has done a decent job, which resulted in decrease in error after every epoch
# and also error was also low compared to TanH and ReLu,
# but error reduction in sigmoid was slow. 


# In[398]:


network = {'layers':[X.shape[1],256,64,10], 'weights':nn.weights,'layerbias':nn.layers, 'activation':'sigmoid'}


# In[391]:


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()

layers = [1,2,3]
loss = [0.4844551378268118,0.4627703481778489,0.28396687375229823]

ax.plot(layers,loss)
plt.xlabel('No. of layers')
plt.ylabel('MSE loss')
plt.show()



# In[392]:


plt.cla()
plt.clf()
plt.close()


# In[394]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
x = range(0,17)
ax.plot(x,error_set)
plt.xlabel('Epochs')
plt.ylabel('MSE loss')
plt.show()


# In[423]:


import pickle
with open(r"network.pickle", "wb") as output_file:
    pickle.dump(network, output_file)


# In[400]:


df = pd.read_csv("../Apparel/apparel-test.csv")
testdata = np.asarray(df)/255


# In[413]:


preds = prediction(nn,X)


# In[416]:


df = pd.DataFrame.from_records(preds)


# In[422]:


df.to_csv("../output_data/2018900060_prediction.csv", sep=',',index=False)


# In[ ]:




