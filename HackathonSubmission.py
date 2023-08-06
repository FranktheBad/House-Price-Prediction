#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the necessary packages

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge


# In[2]:


data=pd.read_csv('Downloads\Housing_dataset_train.csv')
data.head(10)


# In[3]:


data.shape


# In[4]:


data['loc'].unique()


# In[5]:


data['title'].unique()


# ## Exploratory Data Analysis
# 

# In[6]:


#Check for missing values and duplicates
data[data.duplicated()]


# In[7]:


data.isna().sum()


# In[8]:


#Visualize the position of the missing values.
import missingno
missingno.matrix(data)


# In[9]:


data.describe()


# In[10]:


#Create visualizations to understand the effect of the variables on the target i.e price.
group1=data.drop(columns=['bathroom','bedroom','parking_space','ID']).groupby(['loc','title'])
group1.mean().unstack()['price'].plot.bar(figsize=(30,30),subplots=True)


# In[12]:


group1.mean().sort_values(by='price')


# In[13]:


group1.mean().unstack()['price'].plot.line(figsize=(30,30),subplots=True)


# In[14]:


#check the relationship of numerical variable on the price
group2=data.drop(columns=['ID','loc','title','bathroom','parking_space']).groupby('bedroom')
group2.mean().unstack()['price'].plot.bar(figsize=(5,3))


# In[15]:


group3=data.drop(columns=['ID','loc','title','bedroom','parking_space']).groupby('bathroom')
group3.mean().unstack()['price'].plot.bar(figsize=(5,3))


# In[16]:


group4=data.drop(columns=['ID','loc','title','bathroom','bedroom']).groupby('parking_space')
group4.mean().unstack()['price'].plot.bar(figsize=(5,3))


# In[17]:


group5=data.drop(columns=['ID','loc','parking_space','bathroom','bedroom']).groupby('title')
group5.mean().unstack()['price'].plot.bar(figsize=(5,3))


# In[18]:


group6=data.drop(columns=['ID','title','parking_space','bedroom','bathroom']).groupby('loc')
group6.mean().unstack()['price'].plot.bar(figsize=(10,7))


# In[19]:


group6.mean().sort_values(by='price', ascending=False)


# ## Data Cleaning

# In[20]:


#drop null values of categorical features
data=data.dropna(subset=['loc','title'])
data.isna().sum()


# In[21]:


#drop rows with 2 or more missing numerical features.
data = data[data.isna().sum(axis=1) < 2]
data.isna().sum()


# In[22]:


#fill the missing values by the mean of the respective columns/features.
meanbed=round(data['bedroom'].mean())
data['bedroom'].fillna(meanbed, inplace=True)
meanbath=round(data['bathroom'].mean())
data['bathroom'].fillna(meanbath, inplace=True)
meanpark=round(data['parking_space'].mean())
data['parking_space'].fillna(meanpark, inplace=True)


# In[23]:


#Verify filling all missing values
data.isna().sum()


# In[24]:


data.shape


# In[25]:


data.head(10)


# In[26]:


#reset the index of the dataframe
data.reset_index(drop=True, inplace=True)
data.head()


# ## Data Preprocessing
# We perform one hot ecoding on the categorical variables/features (i.e loc and title), in order to make each of the unique entries of these columns a feature that take a binary entry.

# In[27]:


#Writing the One Hot Encoding function
def Create_Expanded_dataset(dataframe, col1, col2):
    columns=[col1, col2]
    for col in columns:
        encoder=OneHotEncoder()
        encarray=encoder.fit_transform(dataframe[[col]]).toarray()
        col_array=encoder.categories_
        col_labels=np.array(col_array).flatten()
        onehotvec=pd.DataFrame(encarray, columns=col_labels)
        dataframe= pd.concat([dataframe, onehotvec], axis=1)
    return dataframe


# In[28]:


expanded_data=Create_Expanded_dataset(data, 'loc','title')
expanded_data.shape


# In[29]:


expanded_data.head()


# In[30]:


#Drop the categorical features
expanded_data.drop(['ID','loc','title'], axis=1, inplace=True)


# In[31]:


expanded_data.head()


# In[32]:


#Extract the X and Y train dataset
X_train=expanded_data.drop('price', axis=1)
y_train=expanded_data['price']


# In[33]:


#Prepare the Test dataset for prediction
test_data=pd.read_csv('Downloads\Housing_dataset_test.csv')
X_test=Create_Expanded_dataset(test_data,'loc','title')
X_test=X_test.drop(['loc','title','ID'], axis=1)
X_test.head()


# ## Model Building
# Here, I first perform a polynomial transformation on the features before carrying out a Kernel Ridge Regression.

# In[34]:


#Convert dataframes to numpy arrays
X_train=X_train.to_numpy()
y_train=y_train.to_numpy()
X_test=X_test.to_numpy()


# In[35]:


#Polynomial transformation of train and test features
poly=PolynomialFeatures(degree=2)

X_train=poly.fit_transform(X_train)
X_test=poly.fit_transform(X_test)


# In[36]:


#Kernel Regression
kernreg=KernelRidge()
kernreg.fit(X_train,y_train)


# In[37]:


predictions=kernreg.predict(X_test)


# In[38]:


#Create Predictions dataframe
pd.options.display.float_format = '{:.2f}'.format
price_dict={'ID':test_data['ID'], 'price':predictions}
price_data=pd.DataFrame(price_dict)
price_data


# In[39]:


#Save the Dataframe
price_data.to_csv('HousePricing.csv', index=False)


# In[ ]:




