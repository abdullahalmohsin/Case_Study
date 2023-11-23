#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset = pd.read_csv(r"C:\Users\Abdullah Al Mohsin\Desktop\python new\missdata.csv")
dataset


# In[2]:


import pandas as pd
dataset = pd.read_csv(r"C:\Users\Abdullah Al Mohsin\Desktop\python new\missdata.csv")
dataset.set_index("Month",inplace = True)
dataset


# In[3]:


newdataset = dataset.interpolate()
newdataset


# In[ ]:




