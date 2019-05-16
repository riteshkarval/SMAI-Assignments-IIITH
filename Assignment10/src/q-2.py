#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np


# In[30]:


seq_chars = ['A','C','G','T']
emmition_prob = {'E':[0.25,0.25,0.25,0.25],
                 '5':[0.05, 0, 0.95, 0],
                 'I':[0.4, 0.1, 0.1, 0.4]}
transition_prob = {'^':{'E':1},
                   'E':{'E':0.9, '5':0.1},
                   '5':{'I':1},
                   'I':{'I':0.9,'$':0.1}}


# In[31]:


sequence = list("CTTCATGTGAAAGCAGACGTAAGTCA")
state_path = list("EEEEEEEEEEEEEEEEEE5IIIIIII$")


# In[32]:


prob = transition_prob['^']['E']


# In[33]:


for i in range(len(sequence)):
    st_0 = state_path[i]
    st_1 = state_path[i+1]
    n = sequence[i]
    
    ep = emmition_prob[st_0][seq_chars.index(n)]
    tp = transition_prob[st_0][st_1]
    prob = prob *ep * tp


# In[34]:


print(np.log(prob))

