#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# In[2]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, RNN


# In[3]:


data = pd.read_csv('../input_data/GoogleStocks.csv')
data = data.drop('date',1)


# In[4]:


data = data.iloc[1:]
data['avg'] = data[['low', 'high']].mean(axis=1)


# In[5]:


data.head()


# In[6]:


sc = MinMaxScaler(feature_range = (0, 1))
scaled_data = sc.fit_transform(data)


# In[7]:


scaled_data[0]


# In[8]:


def RNN2(cells,timesteps):
    X_train = []
    y_train = []
    print('RNN cells:',cells,' Time_Steps:',timesteps, 'layers:2')
    for i in range(timesteps, 755):
        X_train.append([scaled_data[i-timesteps:i, 5],scaled_data[i-timesteps:i, 1]])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.asarray(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[2], 2))
    model = Sequential()
    model.add(GRU(units = cells, return_sequences = True, input_shape = (X_train.shape[1], 2)))
    model.add(GRU(units = cells))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs = 10, batch_size = 32, verbose=0)
    score = model.evaluate(X_train, y_train, batch_size=32, verbose=0)
    return score


# In[9]:


def RNN3(cells,timesteps):
    X_train = []
    y_train = []
    print('RNN cells:',cells,' Time_Steps:',timesteps, 'layers:3')
    for i in range(timesteps, 755):
        X_train.append([scaled_data[i-timesteps:i, 5],scaled_data[i-timesteps:i, 1]])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[2], 2))
    model = Sequential()
    model.add(GRU(units = cells, return_sequences = True, input_shape = (X_train.shape[1], 2)))
    model.add(GRU(units = cells, return_sequences = True))
    model.add(GRU(units = cells))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs = 10, batch_size = 32, verbose=0)
    score = model.evaluate(X_train, y_train, batch_size=32, verbose=0)
    return score


# In[10]:


cells2 = [30,50,80]
timesteps2 = [20,50,75]
cells3 = [30,50,80]
timesteps3 = [20,50,75]


# In[11]:


loss = []
accuracy = []
combination = 1
for cell in cells2:
    for timestep in timesteps2:
        print('Combination:',combination)
        score = RNN2(cell,timestep)
        print('loss:',score[0],' accuracy:',score[1],'\n')
        loss.append(score[0])
        accuracy.append(score[1])
        combination += 1
for cell in cells3:
    for timestep in timesteps3:
        print('Combination:',combination)
        score = RNN2(cell,timestep)
        print('loss:',score[0],' accuracy:',score[1],'\n')
        loss.append(score[0])
        accuracy.append(score[1])
        combination += 1


# In[12]:


print(loss)


# In[13]:


print(accuracy)


# In[ ]:




