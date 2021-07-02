#!/usr/bin/env python
# coding: utf-8

# ## Sound Recognition Using Neural Network

# In[3]:


get_ipython().system('pip install librosa #is a lib for music & sound analysis.provides building blocks to create music info retrieval systems.')


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


filename='/home/shamli/Downloads/Datasets/UrbanSound8K/dog_bark.wav'


# In[7]:


import IPython.display as ipd # helps us to display some of the graphs
import librosa
import librosa.display


# In[8]:


get_ipython().system('dir')


# In[16]:


### Dog Sound
plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename) #gives 2 info:data,sample_rate
librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# - when we are reading this info with the help of librosa,we are reading the signals with the sample rate of 22050.plays very imp role.

# In[17]:


sample_rate #how many times per second a sound is sampled.for audio CDs is 44.1kilohertz)


# In[19]:


from scipy.io import wavfile as wav
wave_sample_rate, wave_audio=wav.read(filename) #if we read wavfile it will give 2info: wave_sample_rate,wave_audio


# In[20]:


wave_sample_rate


# In[21]:


wave_audio


# In[22]:


import pandas as pd

metadata=pd.read_csv('/home/shamli/Downloads/Datasets/UrbanSound8K/metadata/UrbanSound8K.csv')
metadata.head(10)


# - This information says that this particular file is basically present in which folders.

# - To find out how many values are present whether it is balanced/imbalanced dataset.

# In[23]:


### Check whether the dataset is imbalanced
metadata['class'].value_counts()


# - this info is good enough,we know that dataset is imbalanced itself.

# In[26]:


### Sound
filename='/home/shamli/Downloads/Datasets/UrbanSound8K/audio/fold1/15564-2-0-0.wav'
plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename) #gives 2 info:data,sample_rate
librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)


# In[ ]:




