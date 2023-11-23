#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\Abdullah Al Mohsin\Desktop\python new\missdata.csv")
plt.plot(df.Month, df.Quantity)
plt.xticks(rotation=90)


# In[3]:


df = pd.read_csv(r"F:\Ruet academic\6th Semester\6th Sem Myself\Case Study\Jupyter\new_full_data.csv")
plt.plot(df.Month, df.Quantity)
plt.xticks(rotation=90)


# In[4]:


result = adfuller(df.Quantity.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[5]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.Quantity); axes[0, 0].set_title('Original Series')
plot_acf(df.Quantity, ax=axes[0, 1])
# 1st Differencing
axes[1, 0].plot(df.Quantity.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.Quantity.diff().dropna(), ax=axes[1, 1])
# 2nd Differencing
axes[2, 0].plot(df.Quantity.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.Quantity.diff().diff().dropna(), ax=axes[2, 1])
plt.show()


# In[6]:


fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.Quantity.diff()); axes[0].set_title('1st Differencing')
axes[0].set(ylim=(0,5))
plot_pacf(df.Quantity.diff().dropna(), ax=axes[1])
plt.show()


# In[17]:


fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.Sales.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.Sales.diff().dropna(), ax=axes[1])
plt.show()


# In[18]:


df = pd.read_csv(r"F:\Ruet academic\6th Semester\6th Sem Myself\Case Study\Jupyter\new_full_data.csv")
plt.plot(df.Month, df.Quantity)
plt.xticks(rotation=90)


# In[19]:


result = adfuller(df.Quantity.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[20]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.Quantity); axes[0, 0].set_title('Original Series')
plot_acf(df.Quantity, ax=axes[0, 1])
# 1st Differencing
axes[1, 0].plot(df.Quantity.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.Quantity.diff().dropna(), ax=axes[1, 1])
# 2nd Differencing
axes[2, 0].plot(df.Quantity.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.Quantity.diff().diff().dropna(), ax=axes[2, 1])
plt.show()


# In[23]:


fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.Quantity.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.Quantity.diff().dropna(), ax=axes[1])
plt.show()


# In[ ]:




