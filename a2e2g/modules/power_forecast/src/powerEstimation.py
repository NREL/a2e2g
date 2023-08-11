import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np


# Packages for modeling
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from scipy.stats import multivariate_normal
import math



class PowerEstimation:
    
    
    '''
    This class contains functions for computing the power given a distribution 
    for the atmospheric forecast (speed and direction).
    
    The key idea here is to use FLORIS + NN based correction 
    
    
    In order to train the model, I'm taking in FLORIS power as 
    the input to the Neural Network (NN). The output of this NN is the error between
    ground truth and FLORIS power. Then, we add the NN predicted error back 
    to the FLORIS power. This is our corrected power. 
    
    
    After this, we use the speed and direction probabilistic 
    forecast and find the expected power emperically. This includes mean
    and std of power
    
    
    Here are some csv files generated offline:
        
        i) floris_data:: Floris run on the "actual" data 
             (what actually happened that day)
    
        ii) df_floris_sweep:: To find the expectation, we need FLORIS power for various ws and
            wd values. Given that the farm layout won't change,
             I did this offline. 
    
    
    '''
    
    def __init__(self, floris_data,ground_truth, \
                 ws_mean, wd_mean, ws_std, wd_std, df_floris_sweep
                 ):
        
        
        
        self.floris_data = floris_data  # pre-run floris for training
        self.ground_truth = ground_truth 
        self.error = self.ground_truth - self.floris_data # error btw scada and floris
        
        
        # Forecast
        self.ws_mean = ws_mean 
        self.wd_mean = wd_mean
        self.ws_std = ws_std
        self.wd_std = wd_std
        
        
        # This is needed for the expectation: 
        # Sweep through different ws and wd and find powers for the farm
        self.df_floris_sweep = df_floris_sweep.values
        
        
    def Data(self):
        
        # Given the floris power, predict the error:
            
            
        self.train_features = self.floris_data
        self.train_labels = self.error 
        self.test_features = self.floris_data
        self.test_labels = self.error
        
      
        
        
    def train(self):
        
        
        # Get the training and testing data:
        self.Data()
        
        
        # Normalize units
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(self.train_features))
        ws = np.array(self.train_features)

        ws_normalizer = preprocessing.Normalization(input_shape=[1,])
        ws_normalizer.adapt(ws)
        
        
        
        # Play around with this
        self.ws_model = tf.keras.Sequential([
              normalizer,
              layers.Dense(64, activation='relu'),
              layers.Dense(64, activation='relu'),
              #layers.Dense(64, activation='relu'),
              layers.Dense(30)
          ])
        
        
        
        self.ws_model.summary()



        self.ws_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.01),
            loss='mean_squared_error')


        
        history = self.ws_model.fit(
                self.train_features, self.train_labels,
                epochs=1000,
                # suppress logging
                verbose=1,
                # Calculate validation results on 20% of the training data
                validation_split = 0.2)
        
        return self.ws_model
        
        
    
    
    
    def predict(self, feature):
        
        
        '''
        Given a data point, compute corrected power
        
        '''
        power_pred_farm = self.ws_model.predict(feature).sum(axis = 0)
        power_floris_farm = feature.sum(axis = 0)
        
        
        # return floris + predicted correction
        return power_pred_farm + power_floris_farm
        

    
    def probabilisticPower(self):
        
        '''
        We need to find a way to output the mean and std deviation of the famr power.
        I'm going to find the expected power emperically. i.e.,
        
        E[p] = \sum_i Power[i]*N(mean(ws, wd, TI), std(ws, wd, TI))
        E[(p - E(p))**2] = \sum_i (Power[i] - E(p))**2 * 
                               N(mean(ws, wd, TI), std(ws, wd, TI))
        Leaving out TI due to computational limitations. Add later if required.
    
        
        '''
        ln = 50
        ws_space = np.linspace(6, 25, 50)
        wd_space = np.linspace(0, 360, 50)
        E_p = []
        E_std_p = []
        
        # iterate over time
        for time in range(len(self.ws_mean)):
            
            power = []
            prob = []
            # iterate over ws 
            for i in range(ln):
                # iterate over wd
                for j in range(ln):
                    
                    # first get floris inputs for 
                    print("Time:", time)
                    power.append(np.sum(self.predict(self.df_floris_sweep[i * ln + j])))
                    prob_time = multivariate_normal.pdf(
                        [
                            ws_space[i],
                            wd_space[j]
                        ],
                        mean=[
                            self.ws_mean.iloc[time],
                            self.wd_mean.iloc[time]
                        ],
                        cov=[
                            [self.ws_std.iloc[time], 0],
                            [0, self.wd_std.iloc[time]]
                        ]
                    )

                    prob.append(prob_time)

            # make sure they sum to 1
            prob_norm = prob/sum(prob)
                       
            # print('1: ', power[i] - E_p)
            # print('1.5: ', (power[i] - E_p)**2)
            # print('2: ', prob_norm)
            # print('2.5: ', np.shape(prob_norm))
            # print('3: ', len(prob_norm))
            # print('4: ', np.sum([prob_norm[i]*(power[i] - E_p)**2 for i in range(len(prob_norm))]))
            # print('5: ', np.sqrt(np.sum([prob_norm[i]*(power[i] - E_p)**2 for i in range(len(prob_norm))])))

            # Mean
            E_p.append(np.sum([prob_norm[i]*power[i]\
                            for i in range(len(prob_norm))]))
           
            # Variance
            E_std_p.append(np.sqrt(np.sum([prob_norm[i]*(power[i] - E_p[time])**2\
                                for i in range(len(prob_norm))])))
            
        return E_p, E_std_p
            
                
    
    def kfold_cross_val(self):
        
        
        pass
    
    
    def error_metrics(self):
        
        pass
    
    
    def plot(self):
        
       
        df_truth_total = self.ground_truth.sum(axis=1)
        df_floris_total = self.floris_data.sum(axis=1)
        error_pred_total = self.ws_model.predict(self.floris_data).sum(axis = 1)

        t = range(0,len(self.floris_data))
        plt.figure()
        
        plt.plot(t,df_truth_total, label = 'SCADA')
        plt.plot(t,(df_floris_total +\
                 error_pred_total), label = 'Corrected')
       
        
        plt.plot(t,\
        df_floris_total,\
            label = 'FLORIS')
        
        plt.legend()
        
            
        
        
        
        
            
        
    
        
        
        
        