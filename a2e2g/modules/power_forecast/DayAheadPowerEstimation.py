
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# import floris.tools as wfct
import csv
# import src.Cluster as cluster 
from .src import Cluster as cluster 
from .src import NNFarmModel as NNFarmModel
# import src.powerEstimation as powerEstimation
from .src import powerEstimation as powerEstimation

from sklearn.metrics import mean_squared_error

def probabilisticPowerEstimationDayAhead(site_historical=None, forecast=None, data_directory=None):
    

    # %% Datasets useful for calculations:
        
    # floris for the whole ws wd sweep: 
    '''
    Note: I'm using this to avoid running floris while computing the expectation
    
    '''
    df_floris_sweep = pd.read_csv(data_directory + "/floris_datasets/FLORISfullsweep_1.csv")
    print(df_floris_sweep)
    print(site_historical)
    print(forecast)

  

    # Site ws, wd, TI:
    if site_historical == None:
        df = pd.read_csv(data_directory + "/floris_datasets/example_day_20191130_site.csv")

    print(df)

    # corresponding floris power run for the day of interest (11-30-2019)
    df_floris = pd.read_csv(data_directory + "/floris_datasets/FLORIS_full.csv")

    print(df_floris)
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(df_floris['Power0']*1e-6)
    ax[1].plot(df['75m_WS'])
    
    
    # %%  Get indices in the cluster and the corresponding powers
    
   
    layout = pd.read_csv(data_directory + "/floris_datasets/layout.csv")
    g = cluster.Cluster(layout)
    c = g.getTurbineIndices()                              # indices
    df_floris_cluster = pd.DataFrame(df_floris.iloc[:,c])  # powers
    
    # %% Lets get a trained NN for predicting plant power 
    # 
    
    met_wswd = df[['75m_WS','75m_WD' ]]
    
    
    t = NNFarmModel.GroundTruthData(data_directory + "/floris_datasets/layout.csv")
    
    g = t.getModel(data_directory=data_directory)                # this is a TensorFlow model for plant 
    
    
    powGround = g.predict(met_wswd) # Ground Truth Data

    
    # %% This is the main class to compute the probabilistic power 
    

    ws_forecsat = forecast[0]
    wd_forecsat = forecast[1]
    ws_mean = ws_forecsat['Analog Mean']
    ws_std = ws_forecsat['Analog Stde']
    wd_mean = wd_forecsat['Analog Mean']
    wd_std = wd_forecsat['Analog Stde']
    
    # Reduce array size just for integation work
    ws_mean = ws_mean[0:3]
    ws_std = ws_std[0:3]
    wd_mean = wd_mean[0:3]
    wd_std = wd_std[0:3]

    p = powerEstimation.PowerEstimation(df_floris_cluster*1e-6,
                                                powGround[1:,:]*1e-3,ws_mean,\
                            wd_mean, ws_std, wd_std, df_floris_sweep*1e-6)
        

    p.train()               # Train the NN to predict error
    p.plot()
    
    
    return p.probabilisticPower()   # mean and std power
    
    
    
if __name__=='__main__':
    import pickle
    prob_power = probabilisticPowerEstimationDayAhead()
    print(prob_power)

    pickle.dump(prob_power, open('floris_outputs.p', 'wb'))
    
    
    
    
    
    
    
