#!/usr/bin/env python
# coding: utf-8

# In[6]:


import uvicorn
from fastapi import FastAPI
from BankNote import BankNote
import numpy as np
import pandas as pd
import pickle


# In[7]:


app = FastAPI()
pickle_in = open("random_forest_banknote_model.pkl", "rb")
classifier = pickle.load(pickle_in)


# In[8]:


@app.get('/')
def index():
    return {"message": "The App is up and running!"}


# In[10]:


@app.get('/welcome')
def get_name(name: str):
    return {f"The app is up and running {name}"}


# In[11]:


@app.post('/predict')
def predict(data: BankNote):
    data = data.dict() #converting it to dictionary as data is in JSON format usually
    variance = data["variance"]
    skewness = data["skewness"]
    kurtosis = data["kurtosis"]
    entropy = data["entropy"]
    pred = classifier.predict([[variance, skewness, kurtosis, entropy]])
    if pred > 0.5:
        output = "Fake Note"
    else:
        output = "Legit Note"
    return {"prediction": output}


# In[12]:


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


# In[ ]:




