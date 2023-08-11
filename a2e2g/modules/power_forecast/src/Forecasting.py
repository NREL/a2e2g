#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 15:19:21 2021

@author: svijaysh
"""

'''

This class deifnes functions for forecasting 

'''


import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np


# Packages for modeling
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


class Forecasting:
    
    def __init__(self, floris_data,ground_truth, \
                 
                 shift = 5,\
                 train_end_idx = 1400, test_start_idx = 0, test_end_idx = 1400):
        
        self.floris_data = floris_data
        self.ground_truth = ground_truth
        self.df_error =   ground_truth - floris_data
        self.shift = shift
        self.train_end_idx = train_end_idx
        self.test_start_idx = test_start_idx
        self.test_end_idx  = test_end_idx
        self.shift = shift;
        self.ws_mean = ws_mean
        self.wd_mean = wd_mean
        self.
        
        
        
    def Data(self):
        

        
        self.train_features = self.df_error[:self.train_end_idx]
        self.test_features = self.df_error[self.test_start_idx:self.test_end_idx]
        self.train_labels = self.df_error[0 + self.shift:self.train_end_idx + self.shift]
        self.test_labels = self.df_error[self.test_start_idx + self.shift:\
                                         self.test_end_idx + self.shift]
        
        
        
    def train(self):
        
        self.Data()
        
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(self.train_features))
        ws = np.array(self.train_features)

        ws_normalizer = preprocessing.Normalization(input_shape=[1,])
        ws_normalizer.adapt(ws)
        
        
        self.ws_model = tf.keras.Sequential([
              normalizer,
              layers.Dense(64, activation='relu'),
              #layers.Dense(64, activation='relu'),
              layers.Dense(30)
          ])
        
        self.ws_model.summary()



        self.ws_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.0001),
            loss='mean_squared_error')


        
        history = self.ws_model.fit(
                self.train_features, self.train_labels,
                epochs=500,
                # suppress logging
                verbose=1,
                # Calculate validation results on 20% of the training data
                validation_split = 0.2)
        
        return self.ws_model
        
        
        
    def test(self):
        
        self.error_pred = self.ws_model.predict(self.test_features)
        
        
        
        
        return self.error_pred
    
    
    def probabilisticPower(self):
        
        '''
        We need to find a way to output the mean and std deviation of the famr power.
        I'm going to find the expected power emperically. i.e.,
        
        E[p] = \sum_i Power[i]*pdf(mean(ws, wd, TI), std(ws, wd, TI))
        E[(p - E(p))**2] = \sum_i (Power[i] - E(p))**2 * 
                               pdf(mean(ws, wd, TI), std(ws, wd, TI))
        Leaving out TI due to computational limitations. Add later.
    
        
        '''
        ln = 100
        ws_space = np.linspace(5,25, )
        wd_space = np.linspace(0, 360, ln)
        prob = []; power = []
        fi.calculate_wake()
        for time in range()
        for i in range(ln):

            for j in range(ln):
            
                fi.reinitialize_flow_field(wind_speed = ws_space[i], wind_direction = wd_space[j])
                fi.calculate_wake()
                power.append(fi.get_farm_power())
                prob.append(multivariate_normal.pdf([ws_space[i],wd_space[j]],\
                                                    mean = [ws_mean, wd_], cov = [[1,0],[0,10]]))
                
            
    
    def kfold_cross_val(self):
        
        
        pass
    
    
    def error_metrics(self):
        
        pass
    
    
    def plot(self):
        
        self.test()
        df_truth_total = self.ground_truth.sum(axis=1)
        df_floris_total = self.floris_data.sum(axis=1)
        error_pred_total = self.error_pred.sum(axis = 1)

        t = range(0,len(self.error_pred))
        plt.figure()
        
        plt.plot(t,(df_truth_total[self.test_start_idx+\
                self.shift:self.test_end_idx+ self.shift]), label = 'SCADA')
        plt.plot(t,(df_floris_total[self.test_start_idx+\
                    self.shift:self.test_end_idx+ self.shift] +\
                 error_pred_total), label = 'Corrected')
       
        
        plt.plot(t,\
        df_floris_total[self.test_start_idx+ self.shift:self.test_end_idx\
                        + self.shift],\
            label = 'FLORIS')
        
        plt.legend()
        
            
        
        
        
        
            
        
    
        
        
        
        