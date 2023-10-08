#!/usr/bin/env python
# coding: utf-8

# # BHARAT INTERN
# 
# # NAME-SHAIK TABASSUM SHABANA
# 
# # TASK1-STOCK PREDICTION
#  - IN THIS WE WILL USE THE NSE TATA GLOBAL BEVERAGES DATASET FOR STOCK PREDICTION

# # IMPORT LIBRARIES

# In[1]:


from sklearn.linear_model import LogisticRegression 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[8]:


#Load data from a CSV file
df = pd.read_csv(r"C:\Users\smily\Downloads\nse.csv")
df.head()


# # SHAPE OF DATA

# In[9]:


df.shape


# # GATHERING INFORMATION ABOUT THE DATA

# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.dtypes


# In[13]:


df=df.reset_index()['Close']
df


# In[15]:


df.isnull().sum()


# In[16]:


df


# # PREPROCESSING

# In[17]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(np.array(df).reshape(-1,1))


# In[18]:


print(df)


# # DEFINING TIME STEP AND CREATING TRAINING AND TEST DATASETS ACCORDING TO THE TIMESTAMP

# In[19]:


training_size=int(len(df)*0.75)
test_size=int(len(df))-training_size
train_data, test_data = df[0:training_size,:],df[training_size:len(df),:1]


# In[20]:


training_size,test_size


# In[21]:


train_data,test_data


# # CONVERT AN ARRAY VALUES INTO A DATASET

# In[23]:


def create_feartures(dataset,time_steps=1):
    dataX, dataY =[], []
    for i in range(len(dataset)-time_steps-1):
        a=dataset[i:(i+time_steps),0]
        dataX.append(a)
        dataY.append(dataset[i+time_steps, 0])
    return np.array(dataX), np.array(dataY)


# # RESHAPE INTO X=t,t+2,t+3 and t+4 

# In[25]:


ts=100
x_train, y_train=create_feartures(train_data, ts)
x_test, y_test = create_feartures(test_data,ts)


# In[26]:


print(x_train.shape), print(y_train.shape)


# # TRAIN A LINEAR LIGRESSION MODEL 

# In[36]:


model = LinearRegression()
model.fit(x_train, y_train)


# # MAKE PREDICTIONS

# In[38]:


y_pred = model.predict(x_test)


# # CALCULATING RMSE

# In[39]:


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')


# # PLOT THE ACTUAL VS PREDICTED VALUES FOR TRAINING DATA

# In[44]:


import matplotlib.pyplot as plt
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted (Training)")
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted (Testing)")
plt.tight_layout()
plt.show()


# In[45]:


len(test_data)


# In[46]:


x_input=test_data[209:].reshape(1,-1)
x_input.shape


# # PREDICTING THE VALUES FOR NEXT 100 DAYS

# In[47]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input


# In[57]:


day_new=np.imag(1)
day_pred=np.imag(101)


# In[58]:


len(df)


# # THIS IS THE GRAPH OF ACTUAL VALUES IN LAST 100 DAYS

# In[60]:


df1=df.tolist()
df1=scaler.inverse_transform(df1).tolist()
plt.plot(df1)
plt.plot(df)


# In[ ]:




