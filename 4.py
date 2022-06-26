#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, mean_squared_error

import pandas as pd


# In[ ]:


#  Importing flowers as grayscale and i have used PILLOW for grayscaling instead of OPENCV
# because its syntax is easier


# In[ ]:


# importing daisy flowers


# In[ ]:


def assign_label(img,flower_type):
    return flower_type


# In[ ]:


X=[]
Z=[]
def to_solve(flower_type,DIR):
    for img in os.listdir(DIR):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = Image.open(path)
        img_gray = ImageOps.grayscale(img)
        img_gray = img_gray.resize( (28,28))
#         print(img_gray.shape)
        
        X.append(np.array(img_gray))
        Z.append(str(label))
        


# In[2]:


to_solve('Daisy','flowers\daisy')
to_solve('Dandelion','flowers\dandelion')
to_solve('Rose','flowers/rose')
to_solve('Sunflower','flowers\sunflower')
to_solve('Tulip','flowers/tulip')


# In[3]:


X[0].shape


# In[4]:


len(X)


# In[ ]:





# In[ ]:





# In[5]:


le=LabelEncoder()
Y=le.fit_transform(Z)
X=np.array(X)
X=X/255


# In[6]:


Y


# In[ ]:





# In[7]:


# df3.iloc[columns[-1]]


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:



train_images,test_images , train_labels, test_labels = train_test_split(X,Y, test_size=0.23, random_state=42)


# In[9]:


len(train_images)
# len(x_train) #train images


# In[10]:


len(test_images)
# len(y_train) #train labels


# In[11]:


# train_labels=np.array(str(train_labels))
train_labels
# y_train


# There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:

# In[12]:


test_images.shape
# x_train.shape


# And the test set contains 10,000 images labels:

# In[102]:


len(test_labels)
# len(y_test)


# In[103]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# plt.figure()
# plt.imshow(x_train[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


# In[104]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[105]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(1000, activation='relu'),
#     tf.keras.layers.Dense(900, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    
    tf.keras.layers.Dense(250, activation='relu'),
   
    tf.keras.layers.Dense(5)
#     tf.keras.layers.Softmax()
    
])


# In[106]:


model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[107]:


# train_images.shape
train_labels.shape


# In[108]:



model.fit(train_images, train_labels, epochs=100)


# As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.91 (or 91%) on the training data.

# In[109]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


# In[66]:


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


# In[67]:


predictions = probability_model.predict(test_images)


# In[68]:


predictions


# In[69]:


q=[]
for i in range(len(test_labels)):
    q.append(np.argmax(predictions[i]))
    


# In[72]:


confusion_matrix(test_labels,q)


# In[71]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['False', 'True']); ax.yaxis.set_ticklabels(['False', 'True']);


# In[ ]:





# In[ ]:




