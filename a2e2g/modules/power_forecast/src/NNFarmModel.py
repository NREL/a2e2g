import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean


import tensorflow as tf


# Packages for modeling
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.metrics import mean_squared_error

# import src.Cluster as Cluster 
from . import Cluster as Cluster 

class GroundTruthData:
    
    
    '''
    This class gets SCADA data and fits a NN to it
    
    '''
    def __init__(self, data):
        
        self.df = pd.read_csv(data)
        
        
    def cleanData(self):
        
        self.df = self.df.dropna()
        self.df = self.df[:20000]
        
        
    def constructDataset(self, cluster_str):
        
        self.df_power = self.df[cluster_str]

        df_wswd = self.df[['vmet2windsp50', 'vmet2winddr50']]
        
        
        test_start_idx = 19000
        test_end_idx = 20000

        train_start_idx = 0
        train_end_idx = 19000
        
        self.train_features = np.array(df_wswd.iloc[:train_end_idx])
        self.test_features = np.array(df_wswd.iloc[test_start_idx:test_end_idx])
        
        self.train_labels =  self.df_power.iloc[:train_end_idx, :].copy()
        self.test_labels =  self.df_power.iloc[test_start_idx:test_end_idx, :].copy()
        
        
        
    
    
    def trainNN(self):
        
    
        # Normalization:
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(self.train_features))
        
        ws = np.array(self.train_features)
        ws_normalizer = preprocessing.Normalization(input_shape=[1,])
        ws_normalizer.adapt(ws)
        
        
        self.ws_model = tf.keras.Sequential([
                      normalizer,
                      layers.Dense(64, activation='relu'),
                      #layers.Dense(64, activation='relu'),
                      # layers.Dense(64, activation='relu'),
                      # layers.Dense(64, activation='relu'),
                      # layers.Dense(64, activation='relu'),
                      # layers.Dense(64, activation='relu'),
                      layers.Dense(30)
                  ])

        self.ws_model.summary()
        
        self.ws_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
                loss='mean_absolute_error')
        
        history = self.ws_model.fit(
                self.train_features, self.train_labels,
                epochs=100,
                # suppress logging
                verbose=1,
                # Calculate validation results on 20% of the training data
                validation_split = 0.1)
        
        
        # some plots
        predicted = self.ws_model.predict(self.train_features)
        plt.figure()
        plt.plot(predicted, 'r')
        plt.plot(self.train_labels.values, 'y')
        #print(self.train_labels.values.shape)
        print("Mean squared error is:",mean_squared_error(self.train_labels.values,\
                                                          predicted ))
         
        return self.ws_model
   
        
    def testNN(self):
        
        power_train = self.ws_model.predict(self.train_features)
        
        
        fig, ax = plt.subplots(1)
        #ax[0].plot(self.train_features['vmet2windsp50'])
        ax.plot(power_train*1e-3,'r', label = 'prediction')
        ax.plot(self.train_labels.values*1e-3, label = 'SCADA')
        ax.set_ylabel('Farm Power (MW)')
        
        
        plt.legend()
        
        
        print("Mean squared error is:",mean_squared_error(self.train_labels.values*1e-3,\
                                                          power_train*1e-3 ) )
                
        
    def getModel(self, data_directory=None):
    
        c = Cluster.Cluster(self.df)
        t = c.getCluster()
        
        print("Turbines of interest:", t)
        
        
        '''
        Use these turbines to train a NN mapping ws, wd to P_{turbines}
        
        '''
        # %%
        
        b = GroundTruthData(data_directory + "scada.csv")
        b.cleanData()
        
        
        # %%
        b.constructDataset(t)
        
        
        # %%
        model = b.trainNN()
        
        return model
        


     
        