#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Denuwanhh/neural-network-notebooks/blob/main/Ex_01.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


#Import required libraries
import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


#Check The version of Tensorflow
tf.__version__


# In[3]:


##Data Preprocessing

#Import Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values


# In[4]:


print(x)


# In[5]:


print(y)


# In[6]:


#Encode Data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#Ecode data : Gender
x[:,2] = le.fit_transform(x[:,2])


# In[7]:


print(x)


# In[8]:


#One Hot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#One Hot Encoding: Geo
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))


# In[9]:


print(x)


# In[10]:


#Spliting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print('x_train: ' + str(len(x_train)))
print('x_test: ' + str(len(x_test)))
print('y_train: ' + str(len(y_train)))
print('y_test: ' + str(len(y_test)))


# In[11]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print('Train Data Set : \n')
print(x_train)
print('Test Data Set : \n')
print(x_test)


# In[12]:


#Initiate ANN
ann = tf.keras.models.Sequential()


# In[13]:


#Add input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[14]:


#Add second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[15]:


#Add output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[16]:


#Trainning the ANN
#Compile
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])


# In[17]:


#Train the ANN
ann.fit(x_train, y_train, batch_size=32, epochs=100)


# In[19]:


#Prediction
print(ann.predict(sc.transform([[1, 0, 1, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))


# In[20]:


#Prediction using test data
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),1))


# In[22]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)