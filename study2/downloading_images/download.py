import pandas as pd
import numpy as np
import urllib.request 
import os
import glob

file_paths=[]
path_2022_02 = "/project/ll_774_951/uk_ru/twitter/data/2022-02/*.csv"
for fname in glob.glob(path_2022_02):
    file_paths.append(fname)
path_2022_03 = "/project/ll_774_951/uk_ru/twitter/data/2022-03/*.csv"
for fname in glob.glob(path_2022_03):
    file_paths.append(fname)
path_2022_04 = "/project/ll_774_951/uk_ru/twitter/data/2022-04/*.csv"
for fname in glob.glob(path_2022_04):
    file_paths.append(fname)
file_paths.sort()
#file_paths_1 = file_paths[358:458]
#url_set = {}

for file in file_paths:
    df = pd.read_csv(file,  engine='python', error_bad_lines=False)
    
    media_urls = df[['media_urls']]
    rt_media_urls = df[['rt_media_urls']]
    q_media_urls = df[['q_media_urls']]
    
    download_1 = media_urls.loc[df.media_urls.str.contains('http', na=False)]
    download_2 = rt_media_urls.loc[df.rt_media_urls.str.contains('http', na=False)]
    download_3 = q_media_urls.loc[df.q_media_urls.str.contains('http', na=False)]
    
    download_list_1 = download_1["media_urls"].tolist()
    download_list_2 = download_2["rt_media_urls"].tolist()
    download_list_3 = download_3["q_media_urls"].tolist()
    
    special_char = ",]['"
    downloads_1 = [''.join(x for x in string if not x in special_char) for string in download_list_1]
    downloads_2 = [''.join(x for x in string if not x in special_char) for string in download_list_2]
    downloads_3 = [''.join(x for x in string if not x in special_char) for string in download_list_3]
    
    downloads_all = downloads_1+downloads_2+downloads_3
    downloads_all_set = set(downloads_all)
    
    for url in downloads_all_set:
            try:
                urllib.request.urlretrieve(link, "/project/ll_774_951/uk_ru/twitter/twitter_images/" + os.path.basename(link))
            except:
                pass
        
'''  
    for link in downloads_1:
        try:
            url_set = url_set.append(link)
            for url in url_set:
                if url not in url_set:
                    urllib.request.urlretrieve(link, "/project/ll_774_951/uk_ru/twitter/twitter_images/" + os.path.basename(link))
        except:
            pass
    
    for link in downloads_2:
        try:
            urllib.request.urlretrieve(link, "/project/ll_774_951/uk_ru/twitter/twitter_images/" + os.path.basename(link))
        except:
            pass
    
    for link in downloads_3:
        try:
            urllib.request.urlretrieve(link, "/project/ll_774_951/uk_ru/twitter/twitter_images/" + os.path.basename(link))
        except:
            pass
            
'''
