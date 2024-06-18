#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


data = pd.read_csv("D:\\Jeyashri\\IBM\\Datasets\\Country clusters.csv")
data


# In[3]:


plt.scatter(data['Longitude'],data['Latitude'],color='orange')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


# In[4]:


x = data.iloc[:,1:3] # 1t for rows and second for columns
x


# In[10]:


kmeans = KMeans(3)
kmeans.fit(x)


# In[11]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[12]:


data_with_clusters = data.copy()


# In[13]:


data_with_clusters


# In[14]:


data_with_clusters['C'] = identified_clusters 
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['C'],cmap='rainbow')


# In[36]:


data_with_clusters


# In[8]:


individual_clustering_score=[]
for i in range(1,4):
    kmeans=KMeans(n_clusters=i,init='random',random_state=42)
    kmeans.fit(x)
    individual_clustering_score.append(kmeans.inertia_)
plt.plot(range(1,4),individual_clustering_score)
plt.title("ELBOW")
plt.show()

