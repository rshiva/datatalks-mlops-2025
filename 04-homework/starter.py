#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')


# In[18]:


import pickle
import pandas as pd
import numpy as np


# In[22]:


with open('lin_reg.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[36]:


categorical = ['PULocationID','DOLocationID']

def read_data(filename, year, month):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype(str)
    return df



# In[41]:


year = 2023
month = 3
filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02}.parquet' 
print(filename)
df = read_data(filename,year,month)
# df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')
print(df[:5])


# In[27]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[20]:


std_dev = np.std(y_pred)
print(f"Standard Deviation of predicted duration: {std_dev}")


# In[43]:


df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicated_duration': y_pred

})


# In[45]:


output_file = 'output_file.parquet'

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

