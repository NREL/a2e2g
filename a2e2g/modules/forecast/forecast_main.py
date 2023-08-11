#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:03:23 2021

@author: akumler

Main script for the day-ahead forecast, containing all relevant classes for delivering the forecast. 
The core of the forecast code is contained in 'forecast_backend.py'.
"""

# Step 1: Load in necessary Python libraries.

# import pandas as pd
# import numpy as np
# import glob
# import os
# import subprocess
from sklearn.metrics import *
from math import *
# from scipy import interpolate
# from operational_analysis.toolkits import met_data_processing
# import properscoring as ps
import forecast_backend as anen
import time
import pickle

t0 = time.time()
# Step 2: Set some known parameters for the site of interest.
utc_minus = 6
tz_current = 'US/Central'
tz_adjust = 18
tz_adjust_hires = tz_adjust * 12
filename = ' '
hrrr_files = 'data/hrrr/*.csv*'
hrrr_test_range = pd.date_range(start='2019-11-30', end='2019-11-30', freq='1D').date

# Step 3: Load in observations and current day-ahead forecast.
wind_obs = anen.obs_load(filename=filename)

total_hrrr_wspd, total_hrrr_wdir, total_hrrr_ti, total_obs_wspd, total_obs_wdir, total_obs_ti = anen.hrrr_obs_process(hrrr_files=hrrr_files, 
                                                                                                                      obs_file=wind_obs,
                                                                                                                      utc_minus = utc_minus,
                                                                                                                      tz_current = tz_current,
                                                                                                                      tz_adjust = tz_adjust,
                                                                                                                      tz_adjust_hires = tz_adjust_hires)


# Step 4: Run the analog ensemble forecast for the day of interest.
final_wspd, final_wdir, final_ti = anen.analog_forecast(total_hrrr_wspd = total_hrrr_wspd, 
                                                        total_hrrr_wdir = total_hrrr_wdir, 
                                                        total_hrrr_ti = total_hrrr_ti, 
                                                        total_obs_wspd = total_obs_wspd, 
                                                        total_obs_wdir = total_obs_wdir, 
                                                        total_obs_ti = total_obs_ti,
                                                        hrrr_test_range = hrrr_test_range)



t1 = time.time()
total = t1-t0
print('Time elapsed: ' + str(total))

pickle.dump([final_wspd, final_wdir, final_ti], open('forecast_outputs.p', 'wb'))



