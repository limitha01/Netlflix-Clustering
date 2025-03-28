#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
try:
    df = pd.read_csv('Netflix_Dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Netflix_Dataset.csv not found.")
    exit()

# Select numeric features
numeric_data = df.select_dtypes(include=[np.number]).dropna()

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Train the K-Means model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_data)

# Save the model and scaler
joblib.dump(kmeans, 'netflix_kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")


# In[ ]:




