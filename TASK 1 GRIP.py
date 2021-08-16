#!/usr/bin/env python
# coding: utf-8

# # BY:MANOJ GOWD

# # THE SPARK FOUNDATION¶

# # Task 1:Prediction using Supervised ML

# # The task is to Predict the percentage of an student based on the no. of study hours.This will be done using linear regression using 2 variables.

# # Importing libraries

# In[ ]:


started by importing libraries such as pandas,numpy,scikit


# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression


# # STEP 1:READING DATA

# In[2]:


url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")


# In[3]:


s_data.head(8)


# In[4]:


s_data.describe()


# # Step 2 : Data Visualization

# In[5]:


s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Step 3 : Model Traning

# In[7]:


X = s_data.iloc[:, :-1].values  
Y = s_data.iloc[:, 1].values


# In[8]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                            test_size=0.2, random_state=0) 
regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 
print("MODEL IS TRAINED")


# # Step 4 : Plotting the Line of Regression

# In[9]:


line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# # Step 5 : Making Predictions and Comparing

# In[10]:


print(X_test) # Testing data - In Hours
Y_pred = regressor.predict(X_test) # Predicting the scores


# In[11]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
df


# In[12]:


print('Traning score:', regressor.score(X_train,Y_train))
print('Test Score:', regressor.score(X_test,Y_test))


# In[13]:


df.plot(kind='bar',figsize=(5,5))
plt.grid(which='minor', linewidth='0.5',color='pink')
plt.grid(which='minor', linewidth='0.5', color='blue')
plt.show()


# In[14]:


hours = 9.25
test=np.array([hours])
test=test.reshape(-1,1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Step 6 = Evaluation

# In[15]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(Y_test, Y_pred))


# In[ ]:




