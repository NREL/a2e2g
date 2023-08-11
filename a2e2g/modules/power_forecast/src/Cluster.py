#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 19:19:46 2021

@author: svijaysh
"""
import pandas as pd
import utm
import numpy as np
import matplotlib.pyplot as plt

class Cluster:
    
    def __init__(self, layout, reference = None):
        
        
        self.ref = reference
        self.layout = layout
        self.df_layout = None
        
        
    def EastingNorthing(self):
        
        df_layout = self.layout
  
        # add columns with northing and easting
        df_layout['easting'] = 0.0
        df_layout['northing'] = 0.0
        for i in range(len(df_layout)):
            loc = utm.from_latlon((df_layout.iloc[i]['Latitude']), (df_layout.iloc[i]['Longitude']))
            df_layout['easting'].iloc[i] = loc[0]
            df_layout['northing'].iloc[i] = loc[1]
        
        
        return df_layout
    
    
    def getCluster(self):
        
        df_layout  = self.EastingNorthing()
        self.tower = [] 
        for i in range(len(df_layout)):
            x = df_layout['easting'].iloc[i] - np.min(np.array(df_layout['easting']))
            y = df_layout['northing'].iloc[i] - np.min(np.array(df_layout['northing']))
            if i > 5:
                if  2000 < x < 6000 and y > 6000 :
                    plt.plot(x, y, 'r*')
                    self.tower.append('1Minute_' + df_layout['Tag'][i] + 'kw_avg')

                else:
                    plt.plot(x, y, 'bo')
    
                    
        return self.tower
    
    def getTurbineIndices(self):
        plt.figure()
        df_layout  = self.EastingNorthing()
        idx = []
        for i in range(len(df_layout)):
            x = df_layout['easting'].iloc[i] - np.min(np.array(df_layout['easting']))
            y = df_layout['northing'].iloc[i] - np.min(np.array(df_layout['northing']))
            if i > 5:
                if  2000 < x < 6000 and y > 6000 :
                    plt.plot(x, y, 'r*')
                    idx.append(i)
                    
                else:
                    plt.plot(x, y, 'bo')
    
                    
        return idx
        
        
        
        


        