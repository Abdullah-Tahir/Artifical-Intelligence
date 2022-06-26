#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
from collections import Counter

np.random.seed(1337)


# In[23]:


def euclidian_D(query,X):
        difference = np.array(X) - np.array(query)
        sqrd_diff = np.square(difference)
        sum_sqrd_diff = np.sum(sqrd_diff, axis = 1)
        distance = np.sqrt(sum_sqrd_diff)
        return distance


# In[24]:



df = pd.read_csv("fruit_data_with_colors.txt",delimiter = "\t")
X = np.array(df[["mass","width","height","color_score"]])

K = 4

centroidID = np.random.randint(0,58,(K,))
centroids = X[centroidID]

print(centroidID)
print(centroids)


# In[25]:


X_ = np.delete(X,centroidID, axis = 0)
C = [[],[],[],[]]
C[0].append(centroids[0])
C[1].append(centroids[1])
C[2].append(centroids[2])
C[3].append(centroids[3])


# In[26]:


for i in range(10):
    print('iteriate:',i+1, end = '\n')
    C = [[],[],[],[]]

    for x in X:
        id = np.argmin(euclidian_D(x,centroids))
        C[id].append(x)

    c = np.array(C, dtype = object)

    for x in range(4):
        centroids[x] = np.mean(c[x], axis = 0)
        print(centroids[x])

    print()
    


# In[ ]:





# In[ ]:





# In[ ]:




