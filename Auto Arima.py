#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install pmdarima')


# In[2]:


from pmdarima.arima import auto_arima


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from pmdarima.arima import auto_arima


# In[5]:


Quantity_data = pd.read_csv(r"F:\Ruet academic\6th Semester\6th Sem Myself\Case Study\Jupyter\new_full_data.csv")


# In[6]:


Quantity_data.head()


# In[7]:


#Make sure there are no null values
sns.heatmap(Quantity_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[8]:


print(Quantity_data.dtypes)


# In[9]:


Quantity_data['Month']=pd.to_datetime(Quantity_data['Month'])


# In[10]:


Quantity_data.dtypes


# In[11]:


Quantity_data.head()


# In[12]:


plt.figure(figsize=(12,8))
sns.lineplot(data=Quantity_data, x='Month', y= 'Quantity')


# In[13]:


#Set the index of the Month 
Quantity_data.set_index('Month',inplace=True)


# In[14]:


#Testing for stationarity
from pmdarima.arima import ADFTest
adf_test = ADFTest(alpha = 0.05)
adf_test.should_diff(Quantity_data)

