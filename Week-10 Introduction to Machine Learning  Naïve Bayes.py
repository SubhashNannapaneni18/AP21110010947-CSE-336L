#!/usr/bin/env python
# coding: utf-8

# ### Implement Naïve Bayes classifier for following datasets and evaluate the classification performance. Draw the confusion matrix, compute accuracy, error and other measures as applicable.

# In[23]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[24]:


data = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"]
]


# In[25]:


X = np.array([d[:-1] for d in data])
y = np.array([d[-1] for d in data])


# In[26]:


X_encoded = []
for i in range(X.shape[1]):
    unique_values = np.unique(X[:, i])
    encoding = {val: idx for idx, val in enumerate(unique_values)}
    X_encoded.append([encoding[val] for val in X[:, i]])

X_encoded = np.array(X_encoded).T


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[28]:


clf = CategoricalNB()
clf.fit(X_train, y_train)


# In[29]:


y_pred = clf.predict(X_test)


# In[30]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[31]:


error = 1 - accuracy
print("Error:", error)


# In[32]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[33]:


print("Classification Report:")
print(classification_report(y_test, y_pred))


# #### The Iris dataset

# In[34]:


import numpy as np


# In[35]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[37]:


iris = load_iris()
X = iris.data
y = iris.target


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[39]:


clf = GaussianNB()
clf.fit(X_train, y_train)


# In[40]:


y_pred = clf.predict(X_test)


# In[41]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[42]:


error = 1 - accuracy
print("Error:", error)


# In[43]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[44]:


print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[49]:


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Greens", fmt="d", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




