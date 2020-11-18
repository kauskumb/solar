# Copyright 2020 Kaustubh K 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys as sys
import os as os

if sys.platform.startswith('linux'):
	df = pd.read_csv('../solar/data/Plant_1_Generation_Data.csv')
else:
	df = pd.read_csv('E:\GitHub\solar\data\Plant_1_Generation_Data.csv')
df.describe()


# In[2]:


df.head()


# In[3]:

if sys.platform.startswith('linux'):
	df2 = pd.read_csv('../solar/data/Plant_1_Weather_Sensor_Data.csv')
else:
	df2 = pd.read_csv('E:\GitHub\solar\data\Plant_1_Weather_Sensor_Data.csv')
df2.describe()


# In[4]:


df2.head()


# In[5]:


df.shape


# In[6]:


df2.shape


# In[7]:


from datetime import datetime, timedelta
df2['Datetime'] = pd.to_datetime(df2['DATE_TIME'])
from datetime import datetime, timedelta
df['Datetime'] = pd.to_datetime(df['DATE_TIME'])


# In[22]:


import numpy as np
df3 = pd.concat([df, df2], axis=1)

df3.ffill(axis = 0, inplace=True)

df3.drop(['DATE_TIME','PLANT_ID','SOURCE_KEY','DC_POWER','AC_POWER','TOTAL_YIELD','Datetime'], axis=1,inplace=True)
df3 = df3[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DAILY_YIELD']]
df3


# In[31]:


from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
dataset = df3.to_numpy()

# choose a number of time steps
n_steps = 4
# convert into input/output
X, y = split_sequences(dataset, n_steps)
print(y)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
print(type(X))
print(n_features)
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([[23, 33, 22, 0.22], [20, 22, 21, 0.13], [35, 40, 30, 1.3]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)


# In[32]:
x_input = array([[23, 33, 0.22, 4544], [20, 22, 0.13, 3778], [35, 40, 1.3, 6678]])
print (type(array))






