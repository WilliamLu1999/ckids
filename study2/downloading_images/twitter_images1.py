#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import urllib.request
import glob
import requests
import re
from urllib import request
from urllib.parse import urlparse


# In[2]:


#df_media_urls =df.loc[df.media_urls.str.contains('http',na=False)]


# In[3]:


# create a list to store all file paths
file_paths=[]
# after we loop through the folder and get every csv file, we go inside the csv file and grab links for each csv file
path_2022_02 = "/project/ll_774_951/uk_ru/twitter/data/2022-02/*.csv"
for fname in glob.glob(path_2022_02):
    file_paths.append(fname)
path_2022_03 = "/project/ll_774_951/uk_ru/twitter/data/2022-03/*.csv"
for fname in glob.glob(path_2022_03):
    file_paths.append(fname)
file_paths.sort()
# since we divide the downloading task evenly, so I will only deal with the first 522 csv files and their respective links
file_paths_1 = file_paths[0:522]


# In[4]:


# When downloading images, since there are too many, I decide to check 100 csv for every loop
# create a list to store image links
image_links=[]
c="',][" # there are some bad characters we need to remove
for p in file_paths_1[1:522]:
    df = pd.read_csv(p,error_bad_lines=False,engine='python')
    df_media_urls =df.loc[df.media_urls.str.contains('http',na=False)] # select the one that has 
    a =df_media_urls['media_urls'].tolist() # convert it to list to add the link in the list later
    b=[]
    for i in a:
        for char in c:
            i = i.replace(char,"")
        b.append(re.findall(r'(https?://[^\s]+)', i))
    b=list(np.concatenate(b))
    image_links.append(b)


# In[5]:


flat_image_links = list(np.concatenate(image_links))


# In[6]:


print('url collection success')


# In[ ]:


for link in flat_image_links:
    #print(type(os.path.basename(link)))
    try:
        urllib.request.urlretrieve(link,'/project/ll_774_951/uk_ru/twitter/twitter_images/'+os.path.basename(link))
    except:
        pass


# In[ ]:


print('complete!')


# In[ ]:




