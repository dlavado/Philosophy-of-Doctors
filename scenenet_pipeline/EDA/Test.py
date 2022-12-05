# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:59:02 2020

@author: E353247
"""

import pandas as pd
import numpy as np
# import keras
from copy import deepcopy
import scipy
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
# import urllib3, joblib, datetime
import re, math
import h5py
import collections, pickle
# from sklearn import preprocessing
from datetime import date, timedelta
# from sklearn.cluster import DBSCAN, OPTICS
# from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import path
# import tensorflow as tf

#%% to read .dxf files
import geopandas as gpd
doc = gpd.read_file(r"C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR\C1906413/TSCAN/C1906413.dxf")
geoms = doc[doc.Layer=='01_Tower_string'].geometry
dxf_coord_tower = np.array(geoms[155].xy)

#%%
# planolines = doc[doc.Layer=='01_Tower_string'][doc.geometry.type=='LineString']
# planolines.plot()
#%% to read .las files
from laspy.file import File
inFile0 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR\C1906413\LAS\C1906413_20200204_143316_0.las')  #, mode='r'
inFile1 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR\C1906413/LAS\C1906413_20200204_143316_1.las')  #, mode='r'
inFile2 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR/C1906413/LAS/C1906413_20200204_143316_2.las')  #, mode='r'
inFile3 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR/C1906413/LAS/C1906413_20200204_143316_3.las')  #, mode='r'
inFile4 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR/C1906413/LAS/C1906413_20200204_143316_4.las')  #, mode='r'
inFile5 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR/C1906413/LAS/C1906413_20200204_143316_5.las')
inFile6 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR/C1906413/LAS/C1906413_20200204_143316_6.las')  #, mode='r'
inFile7 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR/C1906413/LAS/C1906413_20200204_143316_7.las')  #, mode='r'
inFile8 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR/C1906413/LAS/C1906413_20200204_143316_8.las')  #, mode='r'
inFile9 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR/C1906413/LAS/C1906413_20200204_143316_9.las')  #, mode='r'
inFile10 = File(r'C:\Users\e353247\OneDrive - EDP\EDP_NEW\BigData\LIDAR/C1906413/LAS/C1906413_20200204_143316_10.las')  #, mode='r'  #, mode='r'

inFiles = [inFile0,inFile1,inFile2,inFile3,inFile4,inFile5,inFile6,inFile7,inFile8,inFile9,inFile10]
#inFiles = [inFile0,inFile1]

#dataset = np.vstack([inFile.x, inFile.y, inFile.z]).transpose()
#dataset.shape

#%% read Las files classified
inFile0_class = File('C:/Users/e353247/OneDrive - EDP/EDP_NEW/BigData/LIDAR/C1906413/TSCAN/TILES/C1906413_000001.las') 
# C:\Users\tommi\EDP\Alex Coronati - BigData\LIDAR\C1906413\TSCAN\TILES
#%% Examining Point Format:
print('Examining Point Format: ')
pointformat = inFile.point_format
for spec in inFile.point_format:
   print(spec.name)
#%%
#Lets take a look at the header also.
print('Examining Header Format: ')
headerformat = inFile.header.header_format
for spec in headerformat:
   print(spec.name)
# #%%    
    
# I = inFile.Classification == 2

# points = inFile.points[I]
# print(points)
# print(inFile.intensity)

#%% plot

dataset = np.vstack([inFile0.x, inFile0.y, inFile0.z]).transpose()
xyz = dataset
fig = plt.figure(figsize=[100, 50])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],  marker='.')
# plt.title('Estimated number of cluster: %d' % n_clusters_)
plt.show()
    
    
#%% isolating towers taking the coordinates from the .dxf
idx_a=[]
for inFile in inFiles:
        
    idx=[]
    for i,e in enumerate(dxf_coord_tower[0]):
        idx.append(np.where( (inFile.x-e >= -0.01) & (inFile.x-e <= 0.01) &  
                    (inFile.y-dxf_coord_tower[1][i] >= -0.01) & (inFile.y-dxf_coord_tower[1][i] <= 0.01)))
    
    flat_idx = [item for sublist in idx for item in sublist]            
    idx_a.append(np.concatenate(flat_idx))
   
#%%
idx_b=[]
for inFile in inFiles:   
    idx=[]
    for i,e in enumerate(dxf_coord_tower[0]):
        idx.append(np.where( (abs(inFile.x-e) <= 4) &  
                    ( abs(inFile.y-dxf_coord_tower[1][i]) <= 4)))
    
    flat_idx = [item for sublist in idx for item in sublist]            
    idx_b.append(np.concatenate(flat_idx))
    
#%%
dataset = np.vstack([inFile0.x[idx_b[0]], inFile0.y[idx_b[0]], inFile0.z[idx_b[0]]]).transpose()
xyz = dataset
fig = plt.figure(figsize=[100, 50])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],  marker='.')
# plt.title('Estimated number of cluster: %d' % n_clusters_)
plt.show()

#%% isolating towers points in classified files

a=inFile0_class.x[inFile0_class.raw_classification==15]

b=inFile0_class.y[inFile0_class.raw_classification==15]

c=inFile0_class.z[inFile0_class.raw_classification==15]

dataset = np.vstack([a,b,c]).transpose()
xyz = dataset
fig = plt.figure(figsize=[100, 50])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],  marker='.')
plt.title('plot of classified LiDAR')
plt.show()

#%%
inFile1.intensity.mean()
inFile1.intensity[idx_b[1]].mean()
np.delete(inFile1.intensity, idx_b[1]).mean()
#mark = [vals.index(i) for i in roots]

