#!/usr/bin/env python
# coding: utf-8

# # Train the decision tree with categorical and numerical features. Report precision, recall, f1 score and accuracy.
# Necessary imports and dataset loading
# In[7]:


import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split 
dataset = pd.read_csv("train.csv")  
dataset.keys()
dataset = dataset[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident', 'salary',
       'promotion_last_5years', 'sales', 'left']]

# Function to convert continious data to categorical data.
# In[8]:


def continious_to_categorical(data):
    median = data.median()
    low = data.min()
    high = data.max()
    lmedian = (low+median)/2
    rmedian = (high+median)/2
    temp = []
    for v in data:
        if v < (lmedian+median)/2:
            temp.append('low')
        elif v > (rmedian+median)/2:
            temp.append('high')
        else:
            temp.append('medium')
    return pd.Series(temp)
    

# calling the continious_to_categorical() function for preprocessing
# In[9]:


dataset['satisfaction_level'] = continious_to_categorical(dataset['satisfaction_level'])
dataset['last_evaluation'] = continious_to_categorical(dataset['last_evaluation'])
dataset['number_project'] = continious_to_categorical(dataset['number_project'])
dataset['average_montly_hours'] = continious_to_categorical(dataset['average_montly_hours'])
dataset['time_spend_company'] = continious_to_categorical(dataset['time_spend_company'])

# Function to calculate entropy
# Parameter is column name
# In[10]:


def entropy(column):
    elements,counts = np.unique(column,return_counts = True)
    entropy = 0
    for i in range(len(elements)):
        entropy += -(counts[i]/np.sum(counts))*(np.log2(counts[i]/np.sum(counts)))
    return entropy

# Function to calculate information gain from the column with repect to the target column.
# Parameters are dataset, column name, column name of class
# In[11]:


def InfoGain(data,split,target="left"):
    total_entropy = entropy(data[target])
    vals,counts= np.unique(data[split],return_counts=True)
    Weighted_Entropy = 0
    for i in range(len(vals)):
        weight = counts[i]/np.sum(counts)
        ent = entropy(data.where(data[split]==vals[i]).dropna()[target])
        Weighted_Entropy += weight*ent
    InfoGain = total_entropy - Weighted_Entropy
    return InfoGain

# Function to divide dataset into trian, validate and test sets.
# Size of train dataset is 60% of the data, rest 40% has been divided equally into validation and test set
# In[12]:


def train_validate_test_split(dataset):
    size = len(dataset)
    tsize = int(size*0.6)
    vsize = int(size*0.8)
    training_data = dataset.iloc[:tsize].reset_index(drop=True)
    validation_data = dataset.iloc[tsize:vsize].reset_index(drop=True)
    testing_data = dataset.iloc[vsize:].reset_index(drop=True)
    return training_data,validation_data,testing_data

# Function to create the tree, it uses the ID3(https://en.wikipedia.org/wiki/ID3_algorithm) decision tree training algorithm.
# Prameters to the function are data(subdata if not root), data, column names in dataset, 
# target label, parent of curent node.

# In[13]:


def createtree(subdata,data,attributes,label="left",parent = None):
    if len(np.unique(subdata[label])) <= 1:
        return np.unique(subdata[label])[0]
    elif len(subdata)==0:
        return np.unique(data[label])[np.argmax(np.unique(data[label],return_counts=True)[1])]  
    elif len(attributes) ==0:
        return parent
    else:
        parent = np.unique(subdata[label])[np.argmax(np.unique(subdata[label],return_counts=True)[1])]
        item = [InfoGain(subdata,attribute,label) for attribute in attributes] 
        selected_attribute_index = np.argmax(item)
        selected_attribute = attributes[selected_attribute_index]
        tree = {selected_attribute:{}}
        attributes = [i for i in attributes if i != selected_attribute]
        for value in np.unique(subdata[selected_attribute]):
            value = value
            sub_data = subdata.where(subdata[selected_attribute] == value).dropna()
            subtree = createtree(sub_data,dataset,attributes,label,parent)
            tree[selected_attribute][value] = subtree
        return(tree)

# Function to predict the outcome of sample row. 
# parameters are sample(row of attributes), tree(build from create tree function)
# In[14]:


def predict(sample,tree,default = 1):
    for key in list(sample.keys()):
        if key in list(tree.keys()):
            try:
                prediction = tree[key][sample[key]] 
            except:
                return default
            prediction = tree[key][sample[key]]
            if isinstance(prediction,dict):
                return predict(sample,prediction)
            else:
                return prediction

# Function to handle divide by zero error
# In[15]:


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

# Function to calculate the statistics on the data.
# it prints the classification error, accuracy, precision, recall and F1 score
# on the data provided.
# In[16]:


def stats(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    TP,TN,FP,FN = 0,0,0,0
    for i in range(len(data)):
        if predicted["predicted"].iloc[i] == 0.0:
            if data['left'].iloc[i] == 0:
                TN += 1
            else:
                FN += 1
        else:
            if data['left'].iloc[i] == 0:
                FP += 1
            else:
                TP += 1
    classification_error = safe_div((FP+FN),(TP+FP+TN+FN)) 
    accuracy = safe_div((TP+TN),(TP+FP+TN+FN)) 
    recall = safe_div(TP,(TP+FN))
    precision = safe_div(TP,(TP+FP))
    f1_score = safe_div(2,(safe_div(1,precision))+safe_div(1,recall))
    #print(TP,TN,FP,FN)
    print("Classification error:",classification_error)
    print("Accuracy:",accuracy)
    print("Recall:",recall)
    print("Precision:",precision)
    print("F1 Score:",f1_score)

# Main Function
# In[17]:


if __name__ == "__main__":
    training_data,validation_data,testing_data = train_validate_test_split(dataset)
    tree = createtree(training_data,training_data,training_data.columns[:-1])
    print("Performance on training data")
    stats(training_data,tree)
    print("\nPerformance on validation data")
    stats(validation_data,tree)
    print("\nPerformance on testing data")
    stats(testing_data,tree)


# In[ ]:




