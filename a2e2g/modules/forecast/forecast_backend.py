#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:46:26 2021

@author: akumler

This is the core day-ahead forecast code, upon which 'forecast_main.py' calls from.
"""

import pandas as pd
import numpy as np
import glob
import os
import subprocess
from sklearn.metrics import *
from math import *
from scipy import interpolate
from operational_analysis.toolkits import met_data_processing
import properscoring as ps

def obs_load(filename):
    """
    Load in the historical library of hub-height wind observations.
    
    Parameters
    ----------
    filename: 'String'
        Filename of hub-height wind observations. Should contain wind speed,
        wind direction, and turbulence intensity (TI). Index will be a date-
        time column.
        
    Returns
    ----------
    wind_obs: Pandas DataFrame
        A Pandas DataFrame that contains wind speed, wind direction, and 
        turbuluence intensity (TI) at hub-height.
    """
    # Read in the wind observations.
    wind_obs = pd.read_csv(filename, index_col='Unnamed: 0')
    wind_obs.index = pd.DatetimeIndex(wind_obs.index)
    
    return wind_obs

def hrrr_obs_process(hrrr_files, obs_file, utc_minus, tz_current, tz_adjust,
                     tz_adjust_hires, day='2019-11-30'):
    """
    Load in the historical library of hub-height wind forecasts from HRRR.
    
    Parameters
    ----------
    filename: 'String'
        Filename of hub-height wind forecasts. Should only contain wind speed 
        initially, but wind direction and and turbuluence intensity are 
        calculated in this module.
        
    Returns
    ----------
    wind_hrrr: Pandas DataFrame
        A Pandas DataFrame that contains wind speed, wind direction, and 
        turbuluence intensity at 80 meters.
    """
    # Gather daily HRRR files to process and combine.
    
    hrrr_files = np.sort(glob.glob(hrrr_files))
    
    hrrr_forecast_range = pd.date_range(start='2019-07-01 0' + str(utc_minus) + ':00',
                                   end=day+' 0' + str(utc_minus) + ':00')
    obs_time_range = pd.date_range(start='2019-07-01', end=day)
    total_hrrr_windspeed = pd.DataFrame()
    total_hrrr_winddirec = pd.DataFrame()
    total_hrrr_turbinten = pd.DataFrame()
    total_obs_windspeed = pd.DataFrame()
    total_obs_winddirec = pd.DataFrame()
    total_obs_turbinten = pd.DataFrame()
    count = 0
    used = 0
    skipped = 0
    
    print('Processing HRRR and Observation files...')
    for file in hrrr_files:
        #print('Processing file: ' + file)
        
        try:
            # Need to account for DST somehow.
            current_day = pd.date_range(start=hrrr_forecast_range[count], periods=37, freq='1h')
            current_hires_obs = pd.date_range(start=obs_time_range[count], periods=288, freq='5min')
    
            current_hrrr = pd.read_csv(file)
            current_hrrr = current_hrrr[['Wind', 'U_wind', 'V_wind']]
            current_hrrr.index = pd.DatetimeIndex(current_day)
    
            #Interpolate HRRR to 5-min.
            upsampled_hrrr = current_hrrr.resample('5min')
            current_hires_hrrr = upsampled_hrrr.interpolate(method='cubic')
    
            # Calculate TI.
            hrrr_ti_10min = current_hires_hrrr['Wind'].rolling(2).std() / current_hires_hrrr['Wind'].rolling(2).mean()
            hrrr_ti_30min = current_hires_hrrr['Wind'].rolling(6).std() / current_hires_hrrr['Wind'].rolling(6).mean()
    
            # Convert downscaled u and v components to wind direction.
            current_hires_hrrr['Wind_dir'] = met_data_processing.compute_wind_direction(current_hires_hrrr['U_wind'], current_hires_hrrr['V_wind'])
    
            # Add on 10 and 30-min TI.
            current_hires_hrrr = pd.concat([current_hires_hrrr, hrrr_ti_10min], axis=1)
            #current_hires_hrrr = pd.concat([current_hires_hrrr, hrrr_ti_30min], axis=1)
            current_hires_hrrr.columns = ['Wind', 'U_wind', 'V_wind', 'Wind_dir', 'TI_10min']
    
            # Grab just day-ahead forecast. Fill the rest later with analog ensemble.
            current_hires_hrrr = current_hires_hrrr.iloc[tz_adjust_hires:]
    
            # Combine the forecasts and analogs to their own data frames.
            total_hrrr_windspeed = pd.concat([total_hrrr_windspeed, current_hires_hrrr['Wind']], axis=0)
            total_hrrr_winddirec = pd.concat([total_hrrr_winddirec, current_hires_hrrr['Wind_dir']], axis=0)
            total_hrrr_turbinten = pd.concat([total_hrrr_turbinten, current_hires_hrrr['TI_10min']], axis=0)
    
            current_obs = obs_file.reindex(current_hires_obs)
            current_obs = current_obs[['75m_WS', '75m_WD', '75m_TI_10min', '75m_TI_30min']]
            total_obs_windspeed = pd.concat([total_obs_windspeed, current_obs['75m_WS']], axis=0)
            total_obs_winddirec = pd.concat([total_obs_winddirec, current_obs['75m_WD']], axis=0)
            total_obs_turbinten = pd.concat([total_obs_turbinten, current_obs['75m_TI_10min']], axis=0)
            used += 1
        
        except:
            skipped += 1
            print('No observations for this day: Skipping entirely.')
            continue
            
        
        
        
        count += 1
    print('Files used: {0}. Files skipped: {1}.'.format(used, skipped))
        
    # Give the new data frames uniform index and column names.
    total_hrrr_windspeed[total_hrrr_windspeed < 0] = 0
    total_hrrr_winddirec[total_hrrr_winddirec < 0] = 0
    total_hrrr_turbinten[total_hrrr_turbinten < 0] = 0
    total_obs_windspeed[total_obs_windspeed < 0] = 0
    total_obs_turbinten[total_obs_turbinten < 0] = 0
    total_obs_winddirec[total_obs_winddirec < 0] = 0

    # Make sure all the columns, etc. are in order.
    total_hrrr_ti = pd.DataFrame()
    total_obs_ti = pd.DataFrame()
    total_hrrr_wspd = pd.DataFrame()
    total_obs_wspd = pd.DataFrame()
    total_hrrr_wdir = pd.DataFrame()
    total_obs_wdir = pd.DataFrame()
    
    # Go through each date to make the data frames nice for the current code setup.
    the_dates = total_hrrr_turbinten.index.map(lambda t: t.date()).unique()
    for date in the_dates:
        # HRRR
        temp_hrrr_ti = total_hrrr_turbinten[0][total_hrrr_turbinten.index.date == date]
        # print(temp_hrrr_ti)
        # lkj
        temp_hrrr_ti.reset_index(inplace=True, drop=True)
        # print(temp_hrrr_ti)
        # lkj
        
        temp_hrrr_wspd = total_hrrr_windspeed[0][total_hrrr_windspeed.index.date == date]
        temp_hrrr_wspd.reset_index(inplace=True, drop=True)
        
        temp_hrrr_wdir = total_hrrr_winddirec[0][total_hrrr_winddirec.index.date == date]
        temp_hrrr_wdir.reset_index(inplace=True, drop=True)
        
        # OBS
        temp_obs_ti = total_obs_turbinten[0][total_obs_turbinten.index.date == date]
        temp_obs_ti.reset_index(inplace=True, drop=True)
        
        temp_obs_wspd = total_obs_windspeed[0][total_obs_windspeed.index.date == date]
        temp_obs_wspd.reset_index(inplace=True, drop=True)
        
        temp_obs_wdir = total_obs_winddirec[0][total_obs_winddirec.index.date == date]
        temp_obs_wdir.reset_index(inplace=True, drop=True)
        
        total_hrrr_ti = pd.concat([total_hrrr_ti, temp_hrrr_ti], axis=1)
        total_obs_ti = pd.concat([total_obs_ti, temp_obs_ti], axis=1)
        
        total_hrrr_wspd = pd.concat([total_hrrr_wspd, temp_hrrr_wspd], axis=1)
        total_obs_wspd = pd.concat([total_obs_wspd, temp_obs_wspd], axis=1)
        
        total_hrrr_wdir = pd.concat([total_hrrr_wdir, temp_hrrr_wdir], axis=1)
        total_obs_wdir = pd.concat([total_obs_wdir, temp_obs_wdir], axis=1)
        
    # Convert pandas index to a list of dates.
    the_dates = the_dates.tolist()
    the_dates = [dater.strftime('%Y-%m-%d') for dater in the_dates]
    
    total_hrrr_ti.columns = the_dates
    total_obs_ti.columns = the_dates
    total_hrrr_wspd.columns = the_dates
    total_obs_wspd.columns = the_dates
    total_hrrr_wdir.columns = the_dates
    total_obs_wdir.columns = the_dates

    # Check for nans, get rid of dates that only have nans. DO THIS FOR BOTH FOR OBSERVATIONS AND HRRR.
    for col in total_obs_wspd:
        #print('Processing: ' + str(col))
        #if (total_obs_wspd[col].isnull().sum() != 0):
        if (total_obs_wspd[col].isnull().sum() == 288):
            #print('Bad data: Deleting.')
            #print(total_obs_wspd[col].isnull().sum())
            del total_hrrr_wspd[col]
            del total_hrrr_ti[col]
            del total_obs_wspd[col]
            del total_obs_ti[col]
            del total_hrrr_wdir[col]
            del total_obs_wdir[col]
        else:
            #print('Data looks good.')
            continue
    
    print('Done processing HRRR and observation files.')
    
    # Combine data for easier use.
    #total_hrrr_data = pd.DataFrame()
    #total_hrrr_data = pd.concat([total_hrrr_data, total_hrrr_wspd])
    #total_hrrr_data = pd.concat([total_hrrr_data, total_hrrr_wdir], axis=1)
    #total_hrrr_data = pd.concat([total_hrrr_data, total_hrrr_ti], axis=1)
    
    #total_obs_data = pd.DataFrame()
    #total_obs_data = pd.concat([total_obs_data, total_obs_wspd])
    #total_obs_data = pd.concat([total_obs_data, total_obs_wdir], axis=1)
    #total_obs_data = pd.concat([total_obs_data, total_obs_ti], axis=1)
    
    #return total_hrrr_data, total_obs_data
    return total_hrrr_wspd, total_hrrr_wdir, total_hrrr_ti, total_obs_wspd, total_obs_wdir, total_obs_ti

def analog_forecast(total_hrrr_wspd, total_hrrr_wdir, total_hrrr_ti, total_obs_wspd, total_obs_wdir, total_obs_ti, day):
    """
    Generate a probabilistic forecast utilizing the analog ensemble method. 

    Parameters
    ----------
    hrrr_wspd : Pandas DataFrame
        Processed 5-minute 80 meter wind speed from HRRR.
    hrrr_wdir : Pandas DataFrame
        Processed 5-minute 80 meter wind direction from HRRR.
    hrrr_ti : Pandas DataFrame
        Processed 5-minute 80 meter turbulence intensity from HRRR.
    obs_wspd : Pandas DataFrame
        Processed 5-minute 80 meter wind speed from the site.
    obs_wdir : Pandas DataFrame
        Processed 5-minute 80 meter wind direction from the site.
    obs_ti : Pandas DataFrame
        Processed 5-minute 80 meter turbulence intensity from the site.

    Returns
    -------
    prob_forecast : Pandas DataFrame
        Probabilistic day-ahead forecast for wind speed, wind direction, and turbulence intensity.

    """
    
    # Now that we have our forecasts and observations organized, we can now take some future forecast and compare it
    # to past forecasts. Whichever has the lowest "RMSE", we pick that respective observation file as one analog.


    # Loop over each example forecast day in July, calculate stats and save them, then go to next day.
    # Create loop to go over data every 6 hours.
    # Calcualte the AnEn metric.
    #hrrr_test_range = pd.date_range(start='2019-11-30', end='2019-11-30', freq='1D').date
    hrrr_test_range = pd.date_range(start=day, end=day, freq='1D').date
    total_mbe_analog = pd.DataFrame()
    total_mae_analog = pd.DataFrame()
    total_rmse_analog = pd.DataFrame()
    total_mbe_forecast = pd.DataFrame()
    total_mae_forecast = pd.DataFrame()
    total_rmse_forecast = pd.DataFrame()
    #total_train_forecasts = pd.DataFrame()
    corresponding_weights = pd.DataFrame()
    daily_std = pd.DataFrame()
    daily_std_ti = pd.DataFrame()
    daily_std_wdir = pd.DataFrame()
    total_frame = pd.DataFrame()
    total_crps = pd.DataFrame()
    data_length = len(total_hrrr_ti)
    total_wind_std = pd.DataFrame()
    total_turb_std = pd.DataFrame()
    total_wdir_std = pd.DataFrame()
    
    #train_len = len(hrrr_files) - 2
    train_len = len(total_hrrr_wspd.columns) - 1
    
    for hrrr_test in hrrr_test_range:
        import matplotlib.pyplot as plt
        print('Processing date: ', hrrr_test)
        rmse_wndspd_df = pd.DataFrame()
        rmse_wnddir_df = pd.DataFrame()
        rmse_turint_df = pd.DataFrame()
        stde_df = pd.DataFrame()
        six_obs = pd.DataFrame()
        six_obs_ti = pd.DataFrame()
        six_obs_wdir = pd.DataFrame()
        
        forecast_range = np.arange(0, data_length-2, 36)
        for hour_range in forecast_range:
            #print(hour_range)
            if (hour_range == 396):
                outer_range = hour_range + 36
            else:
                outer_range = hour_range + 36
            # "train" are historical forecasts. "example" is the current forecast.
            train_windspeed_forecasts = total_hrrr_wspd.iloc[hour_range:outer_range,0:train_len]
            example_windspeed_forecast = total_hrrr_wspd[str(hrrr_test)][hour_range:outer_range]
            train_winddirec_forecasts = total_hrrr_wdir.iloc[hour_range:outer_range,0:train_len]
            example_winddirec_forecast = total_hrrr_wdir[str(hrrr_test)][hour_range:outer_range]
            train_turbinten_forecasts = total_hrrr_ti.iloc[hour_range:outer_range,0:train_len]
            example_turbinten_forecast = total_hrrr_ti[str(hrrr_test)][hour_range:outer_range]
            rmse_df = pd.DataFrame()
            delle_tot = pd.DataFrame()
            delle_tot_wdir = pd.DataFrame()
            for col in train_windspeed_forecasts.columns:
                temp_train_wndspd = train_windspeed_forecasts[col]
                #temp_train_wnddir = train_winddirec_forecasts[col]
                temp_train_turint = train_turbinten_forecasts[col]
                
                # Calculate metric from Delle Monache et al. (2011, 2013) for each variable.
                fcst_diff_wndspd = np.sqrt(np.sum((example_windspeed_forecast - temp_train_wndspd)**2))
                wghts_std_wndspd = 1
                
                fcst_diff_turint = np.sqrt(np.sum((example_turbinten_forecast - temp_train_turint)**2))
                wghts_std_turint = 0.5
                
                # Standard deviation of all past forecasts of a given variable at same lead time.
                stde_wndspd = temp_train_wndspd.std()
                stde_turint = temp_train_turint.std()
                
                # Final Delle Monache metric?
                delle_metric = ((wghts_std_wndspd/stde_wndspd) * fcst_diff_wndspd) + ((wghts_std_turint/stde_turint) * fcst_diff_turint)
                #delle_metric = ((wghts_std_wndspd/stde_wndspd) * fcst_diff_wndspd)
                delle_metric = pd.Series(delle_metric)
                
                temp_train_wnddir = train_winddirec_forecasts[col]
                
                # Need to check that wind direction values are reasonable before calculating stats.
                total_wdir_bias = []
                for count, col in enumerate(temp_train_wnddir.values):
                    temp_diff = np.abs(example_winddirec_forecast - temp_train_wnddir)
                    #temp_diff.reset_index(inplace=True, drop=True)
                    for sub_count, sub_col in enumerate(temp_diff.index):
                        if (temp_diff.loc[sub_col] > 180):
                            #smol_guy = dir_forecast.iloc[count,:].nsmallest(1).index[0]
                            if (example_winddirec_forecast.loc[sub_col] < temp_train_wnddir.loc[sub_col]):
                                temp_ana = example_winddirec_forecast.loc[sub_col]
                                temp_obs = temp_train_wnddir.loc[sub_col]
                                wdir_bias = (temp_ana + 360) - temp_obs
                                total_wdir_bias.append(wdir_bias) #pd.concat([total_wdir_bias, wdir_bias])
                            else:
                                temp_ana = example_winddirec_forecast.loc[sub_col]
                                temp_obs = temp_train_wnddir.loc[sub_col]
                                wdir_bias = (temp_obs + 360) - temp_ana
                                total_wdir_bias.append(wdir_bias) #pd.concat([total_wdir_bias, wdir_bias])
                        else:
                            temp_ana = example_winddirec_forecast.loc[sub_col]
                            temp_obs = temp_train_wnddir.loc[sub_col]
                            wdir_bias = temp_ana - temp_obs
                            total_wdir_bias.append(wdir_bias) #pd.concat([total_wdir_bias, wdir_bias])
    
                total_wdir_bias = np.array(total_wdir_bias)
                
                fcst_diff_wnddir = np.sqrt(np.sum((total_wdir_bias)**2))
                
                wghts_std_wnddir = 1
                
                # Need to calculate the standard deviation differently for wind direction.
                # Need to check that wind direction values are reasonable before calculating stats.
                # YAMARTINO METHOD (1984)
                final_frame_wdircopy = temp_train_wnddir.copy()
                total_sine = final_frame_wdircopy.apply(np.sin)
                derp_sine = total_sine.mean(axis=0)
                total_cosine = final_frame_wdircopy.apply(np.cos)
                derp_cosine = total_cosine.mean(axis=0)
    
                eps = np.sqrt(1 - (derp_sine**2 + derp_cosine**2))
    
                std_wdir = np.arcsin(eps)*(1 + 0.1547*(eps)**3)
                std_wdir = pd.Series(np.degrees(std_wdir))
                
                delle_metric_wdir = (wghts_std_wnddir/std_wdir) * fcst_diff_wnddir
                
                delle_metric_wdir = pd.Series(delle_metric_wdir)
                
                delle_tot = pd.concat([delle_tot, delle_metric], axis=1)
                delle_tot_wdir = pd.concat([delle_tot_wdir, delle_metric_wdir], axis=1)
        
        
            delle_tot.columns = train_windspeed_forecasts.columns
            delle_tot = delle_tot.T
            delle_tot_wdir.columns = train_winddirec_forecasts.columns
            delle_tot_wdir = delle_tot_wdir.T
            
            pre_na = delle_tot.copy()
            # Try to drop periods where there are NaN's for some reason
            delle_tot.dropna(inplace=True)
            delle_tot_wdir.dropna(inplace=True)
            
            post_na = delle_tot.copy()
            # 'six_days_first' helps give the weights
            six_days_first = delle_tot.nsmallest(3, 0)
            six_days_first_wdir = delle_tot_wdir.nsmallest(3, 0)
            temp_weights = (1 / six_days_first) / (np.sum(1 / six_days_first))
            temp_weights_wdir = (1 / six_days_first_wdir) / (np.sum(1 / six_days_first_wdir))
            six_obs_init = total_obs_wspd[six_days_first.index.tolist()][hour_range:outer_range]
            six_obs_init_std = np.std(six_obs_init, axis=1)
            total_wind_std = pd.concat([total_wind_std, pd.Series(six_obs_init_std)])
            six_obs_init_ti = total_obs_ti[six_days_first.index.tolist()][hour_range:outer_range]
            six_obs_init_ti_std = np.std(six_obs_init_ti, axis=1)
            total_turb_std = pd.concat([total_turb_std, pd.Series(six_obs_init_ti_std)])
            six_obs_init_wdir = total_obs_wdir[six_days_first_wdir.index.tolist()][hour_range:outer_range]
            
            six_obs_init_wdir_std = six_obs_init_wdir.copy()
            six_obs_init_wdir_std.reset_index(inplace=True, drop=True)
            
            ### Wind direction standard deviation ###
            total_sine = six_obs_init_wdir_std.apply(np.sin)
            total_sine = total_sine.mean(axis=1)
            total_cosine = six_obs_init_wdir_std.apply(np.cos)
            total_cosine = total_cosine.mean(axis=1)
            
            eps = np.sqrt(1 - (total_sine**2 + total_cosine**2))
            
            std_wdir = np.arcsin(eps)*(1 + 0.1547*(eps)**3)
            std_wdir = std_wdir.apply(np.degrees)#pd.Series(np.degrees(std_wdir))
            
            total_wdir_std = pd.concat([total_wdir_std, std_wdir])
            
            for count, col in enumerate(six_obs_init.columns):
                six_obs_init[col] = six_obs_init[col] * temp_weights.iloc[count, 0]
                six_obs_init_ti[col] = six_obs_init_ti[col] * temp_weights.iloc[count, 0]
                
            for count, col in enumerate(six_obs_init_wdir.columns):
                six_obs_init_wdir[col] = six_obs_init_wdir[col] * temp_weights_wdir.iloc[count, 0]
                
            # Wind speed
            six_obs_init = six_obs_init.sum(axis=1)
            six_obs_init.reset_index(inplace=True, drop=True)
            six_obs = pd.concat([six_obs, six_obs_init], axis=1)
            # TI
            six_obs_init_ti = six_obs_init_ti.sum(axis=1)
            six_obs_init_ti.reset_index(inplace=True, drop=True)
            six_obs_ti = pd.concat([six_obs_ti, six_obs_init_ti], axis=1)
            # Wind direction
            six_obs_init_wdir = six_obs_init_wdir.sum(axis=1)
            six_obs_init_wdir.reset_index(inplace=True, drop=True)
            six_obs_wdir = pd.concat([six_obs_wdir, six_obs_init_wdir], axis=1)
            
            # If this is the end of the forecast period, add the remaining hours of the day based on metric
            # of choice below. This is a substitute method until we get more data.
            # So whatever day the forecast was made , we are looking for corresponding observations in the next day.
            # For example, forecast made on 2019-01-31, we need observations for 2019-02-01.
            if (outer_range == 216):
                # Look at RMSE over entire day for starts.
                #train_turbinten_forecasts = total_hrrr_turbinten.iloc[:,0:total_days]
                train_turbinten_forecasts = total_hrrr_ti.iloc[:,0:train_len]
                example_turbinten_forecast = total_hrrr_ti[str(hrrr_test)][:]
                train_windspeed_forecasts = total_hrrr_wspd.iloc[:,0:train_len]
                example_windspeed_forecast = total_hrrr_wspd[str(hrrr_test)][:]
                train_winddirec_forecasts = total_hrrr_wdir.iloc[:,0:train_len]
                example_winddirec_forecast = total_hrrr_wdir[str(hrrr_test)][:]
                rmse_df = pd.DataFrame()
                rmse_df_wdir = pd.DataFrame()
                for col in train_turbinten_forecasts.columns:
                    temp_train_wndspd = train_windspeed_forecasts[col]
                    temp_train_turint = train_turbinten_forecasts[col]
                
                    # Calculate metric from Delle Monache et al. (2011, 2013) for each variable.
                    fcst_diff_wndspd = np.sqrt(np.sum((example_windspeed_forecast - temp_train_wndspd)**2))
                    wghts_std_wndspd = 1
                
                    fcst_diff_turint = np.sqrt(np.sum((example_turbinten_forecast - temp_train_turint)**2))
                    wghts_std_turint = 0.5
                
                    # Standard deviation of all past forecasts of a given variable at same lead time.
                    stde_wndspd = temp_train_wndspd.std()
                    stde_turint = temp_train_turint.std()
                
                    # Final Delle Monache metric?
                    delle_metric = ((wghts_std_wndspd/stde_wndspd) * fcst_diff_wndspd) + ((wghts_std_turint/stde_turint) * fcst_diff_turint)
                    delle_metric = pd.Series(delle_metric)
                    
                    # Calculate separate wind direction metric.
                    temp_train_wnddir = train_winddirec_forecasts[col]
                    
                    # Need to check that wind direction values are reasonable before calculating stats.
                    total_wdir_bias = []
                    for count, col in enumerate(temp_train_wnddir.values):
                        temp_diff = np.abs(example_winddirec_forecast - temp_train_wnddir)
                        for sub_count, sub_col in enumerate(temp_diff.index):
                            if (temp_diff.loc[sub_col] > 180):
                                if (example_winddirec_forecast.loc[sub_col] < temp_train_wnddir.loc[sub_col]):
                                    temp_ana = example_winddirec_forecast.loc[sub_col]
                                    temp_obs = temp_train_wnddir.loc[sub_col]
                                    wdir_bias = (temp_ana + 360) - temp_obs
                                    total_wdir_bias.append(wdir_bias)
                                else:
                                    temp_ana = example_winddirec_forecast.loc[sub_col]
                                    temp_obs = temp_train_wnddir.loc[sub_col]
                                    wdir_bias = (temp_obs + 360) - temp_ana
                                    total_wdir_bias.append(wdir_bias)
                            else:
                                temp_ana = example_winddirec_forecast.loc[sub_col]
                                temp_obs = temp_train_wnddir.loc[sub_col]
                                wdir_bias = temp_ana - temp_obs
                                total_wdir_bias.append(wdir_bias)
    
                    total_wdir_bias = np.array(total_wdir_bias)
                    
                    fcst_diff_wnddir = np.sqrt(np.sum((total_wdir_bias)**2))
                
                    wghts_std_wnddir = 1
                
                    # Need to calculate the standard deviation differently for wind direction.
                    # Need to check that wind direction values are reasonable before calculating stats.
                    # YAMARTINO METHOD (1984)
                    final_frame_wdircopy = temp_train_wnddir.copy()
                    total_sine = final_frame_wdircopy.apply(np.sin)
                    derp_sine = total_sine.mean(axis=0)
                    total_cosine = final_frame_wdircopy.apply(np.cos)
                    derp_cosine = total_cosine.mean(axis=0)
    
                    eps = np.sqrt(1 - (derp_sine**2 + derp_cosine**2))
    
                    std_wdir = np.arcsin(eps)*(1 + 0.1547*(eps)**3)
                    std_wdir = pd.Series(np.degrees(std_wdir))
                                    
                    delle_metric_wdir = (wghts_std_wnddir/std_wdir) * fcst_diff_wnddir
                    
                    delle_metric_wdir = pd.Series(delle_metric_wdir)
    
                    rmse_df = pd.concat([rmse_df, delle_metric], axis=1)
                    
                    rmse_df_wdir = pd.concat([rmse_df_wdir, delle_metric_wdir], axis=1)
                rmse_df.columns = train_windspeed_forecasts.columns
                rmse_df = rmse_df.T
                
                rmse_df_wdir.columns = train_winddirec_forecasts.columns
                
                rmse_df_wdir = rmse_df_wdir.T
                
                # Try to drop periods where there are NaN's for some reason
                rmse_df.dropna(inplace=True)
                rmse_df_wdir.dropna(inplace=True)
                
                # 'six_days_first' helps give the weights
                six_days_first = rmse_df.nsmallest(3, 0)
                six_days_first_wdir = rmse_df_wdir.nsmallest(3, 0)
                temp_weights = (1 / six_days_first) / (np.sum(1 / six_days_first))
                temp_weights_wdir = (1 / six_days_first_wdir) / (np.sum(1 / six_days_first_wdir))
                six_obs_init = total_obs_wspd[six_days_first.index.tolist()][:]
                six_obs_init_std = np.std(six_obs_init, axis=1)
                six_obs_init_std = six_obs_init_std[outer_range:]
                total_wind_std = pd.concat([total_wind_std, pd.Series(six_obs_init_std)])
                six_obs_init_ti = total_obs_ti[six_days_first.index.tolist()][:]
                six_obs_init_ti_std = np.std(six_obs_init_ti, axis=1)
                six_obs_init_ti_std = six_obs_init_ti_std[outer_range:]
                total_turb_std = pd.concat([total_turb_std, pd.Series(six_obs_init_ti_std)])
                six_obs_init_wdir = total_obs_wdir[six_days_first_wdir.index.tolist()][:]
                
                six_days_init_wdir_std = six_obs_init_wdir[outer_range:]
                six_days_init_wdir_std.reset_index(inplace=True, drop=True)
                
                ### Wind direction standard deviation ###
                total_sine = six_days_init_wdir_std.apply(np.sin)
                total_sine = total_sine.mean(axis=1)
                total_cosine = six_days_init_wdir_std.apply(np.cos)
                total_cosine = total_cosine.mean(axis=1)
                
                #eps = np.sqrt(1 - (derp_sine**2 + derp_cosine**2))
                eps = np.sqrt(1 - (total_sine**2 + total_cosine**2))
                
                std_wdir = np.arcsin(eps)*(1 + 0.1547*(eps)**3)
                std_wdir = std_wdir.apply(np.degrees)#pd.Series(np.degrees(std_wdir))
                
                total_wdir_std = pd.concat([total_wdir_std, std_wdir])
                
                for count, col in enumerate(six_obs_init.columns):
                    six_obs_init[col] = six_obs_init[col] * temp_weights.iloc[count, 0]
                    six_obs_init_ti[col] = six_obs_init_ti[col] * temp_weights.iloc[count, 0]
                    
                for count, col in enumerate(six_obs_init_wdir.columns):
                    six_obs_init_wdir[col] = six_obs_init_wdir[col] * temp_weights_wdir.iloc[count, 0]
                
                # Wind speed
                six_obs_init = six_obs_init.sum(axis=1)
                six_obs_init.reset_index(inplace=True, drop=True)
                # Just grab the last few hours of the day we need.
                final_six_obs = six_obs_init[outer_range:]
                # TI
                six_obs_init_ti = six_obs_init_ti.sum(axis=1)
                six_obs_init_ti.reset_index(inplace=True, drop=True)
                # Just grab the last few hours of the day we need.
                final_six_obs_ti = six_obs_init_ti[outer_range:]
                # Wind direction
                six_obs_init_wdir = six_obs_init_wdir.sum(axis=1)
                six_obs_init_wdir.reset_index(inplace=True, drop=True)
                # Just grab the last few hours of the day we need.
                final_six_obs_wdir = six_obs_init_wdir[outer_range:]
            
        six_new = six_obs.copy()
        six_new_ti = six_obs_ti.copy()
        six_new_wdir = six_obs_wdir.copy()

        # Wind speed
        derp = pd.melt(six_new)
        derp = derp['value']
        # Add remaining part of day from second loop.
        derp = pd.concat([derp, final_six_obs])
        derp.dropna(how='any', inplace=True)
        derp.reset_index(inplace=True, drop=True)
        
        # TI
        derp_ti = pd.melt(six_new_ti)
        derp_ti = derp_ti['value']
        # Add remaining part of day from second loop.
        derp_ti = pd.concat([derp_ti, final_six_obs_ti])
        derp_ti.dropna(how='any', inplace=True)
        derp_ti.reset_index(inplace=True, drop=True)
        
        # Wind direction
        derp_wdir = pd.melt(six_new_wdir)
        derp_wdir = derp_wdir['value']
        # Add remaining part of day from second loop.
        derp_wdir = pd.concat([derp_wdir, final_six_obs_wdir])
        derp_wdir.dropna(how='any', inplace=True)
        derp_wdir.reset_index(inplace=True, drop=True)
        
        # Make a plot to see what the analogs look like compared to the forecast!
        final_frame = pd.DataFrame()
        final_frame_ti = pd.DataFrame()
        final_frame_wdir = pd.DataFrame()
        final_frame = pd.concat([final_frame, derp], axis=1)
        final_frame_ti = pd.concat([final_frame_ti, derp_ti], axis=1)
        final_frame_wdir = pd.concat([final_frame_wdir, derp_wdir], axis=1)
        final_frame = final_frame.mean(axis=1)
        final_frame_ti = final_frame_ti.mean(axis=1)
        #day_std = pd.Series(final_frame.std(skipna=True, axis=0))
        #day_std_ti = pd.Series(final_frame_ti.std(skipna=True, axis=0))
        #day_std_wdir = pd.Series(final_frame_wdir.std(skipna=True, axis=0))
        #daily_std = pd.concat([daily_std, day_std])
        #daily_std_ti = pd.concat([daily_std_ti, day_std_ti])
        #daily_std_wdir = pd.concat([daily_std_wdir, day_std_wdir])
        orig_hrrr = total_hrrr_wspd[str(hrrr_test)]
        orig_hrrr.reset_index(inplace=True, drop=True)
        orig_hrrr_ti = total_hrrr_ti[str(hrrr_test)]
        orig_hrrr_ti.reset_index(inplace=True, drop=True)
        orig_hrrr_wdir = total_hrrr_wdir[str(hrrr_test)]
        orig_hrrr_wdir.reset_index(inplace=True, drop=True)
        final_frame = pd.concat([final_frame, total_wind_std], axis=1)
        final_frame = pd.concat([final_frame, orig_hrrr], axis=1)
        final_frame = pd.concat([final_frame, total_obs_wspd[str(hrrr_test)]], axis=1)
        final_frame_ti = pd.concat([final_frame_ti, total_turb_std], axis=1)
        final_frame_ti = pd.concat([final_frame_ti, orig_hrrr_ti], axis=1)
        final_frame_ti = pd.concat([final_frame_ti, total_obs_ti[str(hrrr_test)]], axis=1)
        total_wdir_std.reset_index(inplace=True, drop=True)
        final_frame_wdir = pd.concat([final_frame_wdir, total_wdir_std], axis=1)
        final_frame_wdir = pd.concat([final_frame_wdir, orig_hrrr_wdir], axis=1)
        final_frame_wdir = pd.concat([final_frame_wdir, total_obs_wdir[str(hrrr_test)]], axis=1)
        final_frame.columns = ['Analog Mean', 'Analog Stde', 'Orig HRRR', 'Observations']
        final_frame_ti.columns = ['Analog Mean', 'Analog Stde', 'Orig HRRR', 'Observations']
        final_frame_wdir.columns = ['Analog Mean', 'Analog Stde', 'Orig HRRR', 'Observations']
        
        forecast_time = pd.date_range(start=hrrr_test_range[0], freq='5min', periods=288)
        final_frame.index = forecast_time
        final_frame_ti.index = forecast_time
        final_frame_wdir.index = forecast_time
        
        return final_frame, final_frame_wdir, final_frame_ti















