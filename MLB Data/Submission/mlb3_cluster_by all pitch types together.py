# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:17:44 2019

@author: Erick
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from kneed import KneeLocator

#%%
"""
THIS SECTION JOINS THE YEARS TOGETHER AND SPLITS FILE ON PITCHER HANDEDNESS
"""

path_list = ['2019', '2018b'] #, '2018a', '2017', '2016', '2015'] 
path = "C:/Baseballcloud/MLB Data/"
df = pd.DataFrame()
for fname in path_list:
   current_csv = pd.read_csv(path + fname + '_3D_spin.csv', header=0)
   df = df.append(current_csv)

hands = ['R'] #, 'L']
for hand in hands:
    # Select Handedness
    current = df[(df['p_throws'] == hand)]
    loc = 'C:/Baseballcloud/MLB Data/{}_2015_2019_w_spin.h5'.format(hand) 
    current.to_hdf(loc, key='train', mode='w')
    
#%%
def get_clust(df, x, i):
    # Rescale the data and grab only relevant input columns
    scaler = MinMaxScaler()
    copy = df[['release_speed', 'pfx_x', 'pfx_z']].copy()
    #copy['RelSpeed'] = copy['RelSpeed'] ** 2
    X = scaler.fit_transform(copy)
    X = pd.DataFrame(X, columns = ['release_speed', 'pfx_x', 'pfx_z'])
    
    # Elbow method
    distortions = []
    K = range(1,7)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X,kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    
    # KneeLocater notes at: https://github.com/arvkevi/kneed/blob/master/notebooks/decreasing_function_walkthrough.ipynb
    kn = KneeLocator(list(K), distortions, S=1.0, curve='convex', direction='decreasing')
    print(kn.knee) # Optimal knee according to elbow method
    
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.plot(K, distortions, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()
    # Use Kmeans to split into clusters
    plt.clf()
    kmeans = KMeans(n_clusters= 30, n_init=100, n_jobs = -1) # You can update the number of clusters you'd like here
    kmeans.fit(X)   
    #y_kmeans = kmeans.predict(X)
    # save the model to disk
    file_loc = 'C:/Baseballcloud/MLB Data/Clusters/'
    filename = '{}kmeans_model_{}_{}.sav'.format(file_loc,x,i)
    pickle.dump(kmeans, open(filename, 'wb'))
    
    # Get the data describing the centroid of each cluster, rescale and save centroid info
    centers = kmeans.cluster_centers_
    centers_ = scaler.inverse_transform(centers)
    center_df = pd.DataFrame(centers_, columns = ['release_speed','pfx_x', 'pfx_z']) 
    #center_df['RelSpeed'] = center_df['RelSpeed'] ** 1/2
    centers_save_location = 'C:/Baseballcloud/MLB Data/Clusters/centers_{}_{}.csv'.format(x,i)
    center_df.to_csv(centers_save_location, index = False)
    
    # Add cluster labels onto original data file and save
    labels = kmeans.labels_
    df['Cluster'] = labels
    data_save_location = 'C:/Baseballcloud/MLB Data/Clusters/data_w_labels_{}_{}.h5'.format( x,i)
    df.to_hdf(data_save_location, key='train', mode='w')
    

    # Plot each combo of input variables and save plots
    plot_save_location = 'C:/Baseballcloud/MLB Data/Clusters/plots_'
    c=df['Cluster'] # Color by cluster label
    
    plt.clf()
    plt.scatter(df['pfx_x'], df['pfx_z'], c=c, s=50, cmap='viridis')
    plt.xlabel('HorzBreak')
    plt.ylabel('InducedVertBreak')
    plt.title('{} '.format(i))
    plt.savefig(plot_save_location + '{}_{}_Breaks.png'.format(x, i))
    plt.show()    
    
    plt.clf()
    plt.scatter(df['release_speed'], df['pfx_x'], c=c, s=50, cmap='viridis')
    plt.xlabel('release_speed')
    plt.ylabel('HorzBreak')
    plt.title('{} '.format(i))
    plt.savefig(plot_save_location + '{}_{}_Speed & Horz.png'.format(x, i))
    plt.show()
    
    plt.clf()
    plt.scatter(df['release_speed'], df['pfx_z'], c=c, s=50, cmap='viridis')
    plt.xlabel('RelSpeed')
    plt.ylabel('InducedVertBreak')
    plt.title('{} '.format( i))
    plt.savefig(plot_save_location + '{}_{}_Speed & Vert.png'.format(x, i))
    plt.show()

    
#%%

#pitches = ['Cutter', 'ChangeUp', 'Splitter', 'Sinker','Fastball', 'Curveball', 'Slider']

#df_left = pd.read_csv('C:/Baseballcloud/MLB Data/L_2015_2019_w_spin.csv')
#df_right = pd.read_csv('C:/Baseballcloud/MLB Data/R_2019_w_spin.csv') #_2015
df_right = pd.read_hdf('C:/Baseballcloud/MLB Data/R_2015_2019_w_spin.h5')

#cols = list(df_left.columns)
#%%
#df_left.dropna(axis = 0, subset = ['release_speed', 'pfx_x', 'pfx_z'], inplace = True)
df_right.dropna(axis = 0, subset = ['release_speed', 'pfx_x', 'pfx_z'], inplace = True)
#%%
#cols = list(df_right.columns)
#print(cols)
get_clust(df_right, 'all', 'R')
#get_clust(df_left, 'all', 'L')
