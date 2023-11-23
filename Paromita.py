#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv(r"F:\Ruet academic\6th Semester\6th Sem Myself\Case Study\Jupyter\new_full_data.csv", parse_dates=['Month'], index_col=['Month'])


# In[4]:


df.head()


# In[6]:


df.plot()


# In[7]:


import statsmodels.api as sm


# In[10]:


from statsmodels.tsa.stattools import adfuller


# In[11]:


adftest=adfuller(df)


# In[12]:


print('pvalue of adfuller test is: ', adftest[1])


# In[13]:


len(df)


# In[36]:


train=df[:25]
test=df[25:]


# In[37]:


import itertools


# In[38]:


p=range(0,8)
q=range(0,8)
d=range(0,2)


# In[39]:


pdq_combination=list(itertools.product(p,d,q))


# In[40]:


pdq_combination


# In[41]:


len(pdq_combination)


# In[42]:


rmse=[]
order1=[]


# In[43]:


for pdq in pdq_combination:
    try:
        model=ARIMA(train,order=pdq).fit()
        pred=model.predict(start=len(train),end=(len(df)-1))
        error=np.sqrt(mean_squared_error(test,pred))
        order1.append(pdq)
        rmse.append(error)
        
    except:
        continue


# In[44]:


results=pd.DataFrame(index=order1,data=rmse, columns=['RMSE'])


# In[45]:


results.head()


# In[46]:


df.head()


# In[47]:


train.head()


# In[48]:


test.head()


# In[49]:


len(train)


# In[50]:


len(test)


# In[ ]:




