#!/usr/bin/env python
# coding: utf-8

# # Visualise training data on a 2-dimensional plot taking one feature (attribute) on one axis and other feature on another axis. Take two suitable features to visualise decision tree boundary (Hint: use scatter plot with differentcolors for each label).

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv("train.csv")


# In[3]:


satisfaction_level = dataset['satisfaction_level']
average_montly_hours = dataset['average_montly_hours']
left = dataset['left']


# In[4]:


xt,yt = [],[]
xf,yf = [],[]


# In[5]:


for i in range(len(left)):
    if left[i] == 1:
        xt.append(satisfaction_level[i])
        yt.append(average_montly_hours[i])
    else:
        xf.append(satisfaction_level[i])
        yf.append(average_montly_hours[i])

# 2D visualixzation of dataset using the "satisfaction level" and "average monthly salary attributes"
# Green points shows the where points(satisfaction level","average monthly salary attributes") are true and red shows the false ones.
# In[13]:


plt.scatter(xf,yf, color='r',s=0.5)
plt.scatter(xt,yt, color='g',s=0.5)
plt.xlabel('Satisfaction Level')
plt.ylabel('Average Montly Hours')
plt.show()


# In[ ]:




