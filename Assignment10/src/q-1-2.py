#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from hmmlearn import hmm
from sklearn.metrics import mean_squared_error


# In[77]:


data = pd.read_csv('../input_data/GoogleStocks.csv')
data = data.drop('date',1)
data = data.iloc[1:]
data['avg'] = data[['low', 'high']].mean(axis=1)


# In[83]:


def HMM(states, timesteps):
    X1_train = []
    X2_train = []
    y_train = []
    print('States:',states,' Timesteps:',timesteps)
    for i in range(timesteps, 755):
        X1_train.append(scaled_data[i-timesteps:i, 5])
        X2_train.append(scaled_data[i-timesteps:i, 1])
        y_train.append(scaled_data[i, 0])
    X1_train,X2_train, y_train = np.asarray(X1_train) ,np.asarray(X2_train), np.array(y_train)
    X = np.column_stack([X1_train, X2_train])
    
    
    remodel = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
    remodel.fit(X)
    hidden_states = remodel.predict(X)
    expected_returns_and_volumes = np.dot(remodel.transmat_, remodel.means_)
    returns_and_volume_columnwise = list(zip(*expected_returns_and_volumes))
    expected_returns = returns_and_volume_columnwise[0]
    predicted_prices = []
    predicted_volumes = []
    for idx in range(755-timesteps):
        state = hidden_states[idx]
        current_price = scaled_data[idx][1]
        predicted_prices.append(current_price + expected_returns[state])
    mse = mean_squared_error(predicted_prices,y_train)
    return mse


# In[84]:

sc = MinMaxScaler(feature_range = (0, 1))
scaled_data = sc.fit_transform(data)
timesteps_list = [20,50,75]
states_list = [4,8,12]
combination = 1
score_list = []
for states in states_list:
    for timesteps in timesteps_list:
        print('Combination:',combination)
        score = HMM(states,timesteps)
        score_list.append(score)
        print('Loss:',score,'\n')
        combination += 1


# In[ ]:




