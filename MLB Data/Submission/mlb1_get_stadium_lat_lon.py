# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:27:56 2019

@author: Erick
"""

import pandas as pd
import json

#%%
# Lat Long info taken from: https://gist.github.com/the55/2155142
# Compare to 2017 listing jpg 
with open("mlb_stadium.json", "r") as read_file:
    data = json.load(read_file)
    
# create dataframe from list of dicts
locator_df = pd.DataFrame(data)

#%%
    
df = pd.DataFrame()
#path_list = ["../trackman data/Trackman By Year/2015_trackman.csv", "../trackman data/Trackman By Year/2016_trackman.csv", "../trackman data/Trackman By Year/2017_trackman.csv", "../trackman data/Trackman By Year/2018_trackman.csv", "../trackman data/Trackman By Year/2019_trackman.csv"]
path_list = ["/2015.csv"] 
#path_list = ["/2019.csv"]
path = "C:/Baseballcloud/MLB Data"
for fname in path_list:
   current_csv = pd.read_csv(path + fname, header=0)
   df = df.append(current_csv)

#print(df.columns)

#%%

# Don't need to do this for every year after initial stad_conversion_key.df has been made

track_stadiums = list(df.home_team.unique())
loc_stadiums = list(locator_df.team.unique())

stad_df = pd.DataFrame()
stad_df["Trackman"] = track_stadiums
stad_df["Locator"] = loc_stadiums

# Download and manually match columns, then reupload
#stad_df.to_csv('stad_conversion_key.csv', index = False)

#%%

stad_df = pd.read_csv('stad_conversion_key.csv')
stadium_info = stad_df.merge(locator_df, how='left', left_on='Locator', right_on='team')
stadium_info.drop(['Locator', 'address'], axis =1, inplace = True)
stadium_info.to_csv('stadium_info.csv', index = False)
