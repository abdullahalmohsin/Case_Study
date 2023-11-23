#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"F:\Ruet academic\6th Semester\6th Sem Myself\Case Study\Jupyter\new_full_data.csv", parse_dates=['Month'], index_col=['Month'])


# In[3]:


df.head()


# In[4]:


df.plot()


# In[5]:


import statsmodels.api as sm


# In[6]:


from statsmodels.tsa.stattools import adfuller


# In[7]:


adftest=adfuller(df)


# In[8]:


print('pvalue of adfuller test is: ', adftest[1])


# In[9]:


len(df)


# In[10]:


train=df[:35]
test=df[35:]


# In[11]:


import itertools


# In[12]:


p=range(0,8)
q=range(0,8)
d=range(0,2)


# In[13]:


pdq_combination=list(itertools.product(p,d,q))


# In[14]:


pdq_combination


# In[15]:


len(pdq_combination)


# In[16]:


rmse=[]
order1=[]


# In[17]:


for pdq in pdq_combination:
    try:
        model=ARIMA(train,order=pdq).fit()
        pred=model.predict(start=len(train),end=(len(df)-1))
        error=np.sqrt(mean_squared_error(test,pred))
        order1.append(pdq)
        rmse.append(error)
        
    except:
        continue


# In[18]:


results=pd.DataFrame(index=order1,data=rmse, columns=['RMSE'])


# In[19]:


results.head()


# In[ ]:




