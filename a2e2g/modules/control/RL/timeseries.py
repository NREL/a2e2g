import pandas as pd
from a2e2g.modules.control_simulation.truth_sim import wrap_to_360
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from pathlib import Path
PKGROOT = str(Path(__file__).resolve().parents[3])
running_on_eagle = True

def GetData(df_wind=None, df_AGC_signal=None):   
    '''
    
    Wind speed: Resolution is 1min
    AGC: Resolution is 4s
    

    '''
    if df_AGC_signal is None:
        # AGC from Elina
        if running_on_eagle:
            df_AGC_signal = pd.read_csv(
                PKGROOT + "/data/floris_datasets/AGC_simple_regonly_for_training.csv"
            )
        else:
            df_AGC_signal = pd.read_csv(
                "./a2e2g/modules/control/datasets/AGC_simple_regonly_for_training.csv"
            )

    if df_wind is None:
        # Speeds from Andrew
        # CHANGE DATA TO "wind_obs_20190801-20200727_v2.csv"
        if running_on_eagle:
            df_wind = pd.read_csv(
                PKGROOT + "/data/floris_datasets/wind_obs_20190801-20200727_v2.csv"
            )
        else:
            df_wind = pd.read_csv(
                "./a2e2g/modules/control/datasets/wind_obs_20190801-20200727_v2.csv"
            )
        df_wind.rename(columns={'Unnamed: 0':'datetime'}, inplace=True)
        df_wind = df_wind[(df_wind.datetime >= '2019-08-01') & 
                          (df_wind.datetime < '2019-09-01')] # Get only August
        # fills NaNs
        df_wind.interpolate(method='polynomial', order=2, inplace=True)
    
    # Data 
    num_days = round(len(df_wind)/24/60)
    num_sec_day = 86400
    interval_agc = 4
    interval_wind = 60

    df_AGC_signal = df_AGC_signal.iloc[:num_days*num_sec_day//interval_agc]
    df_wind = df_wind.iloc[:num_days*num_sec_day//interval_wind]

    time_pow_4s = np.arange(0, num_days*num_sec_day, interval_agc)
    power_4s = df_AGC_signal['Basepoint signal'].to_numpy()
    
    # Upsample wind
    time_1min = np.arange(0, num_days*num_sec_day, interval_wind)
    wind_1min = np.linalg.norm(df_wind[['75m_U', '75m_V']].to_numpy(), axis=1)
    wind_1min_f = interpolate.interp1d(time_1min, wind_1min, kind='quadratic', 
                                       fill_value='extrapolate')
    wind_4s = wind_1min_f(time_pow_4s)
    
    return time_pow_4s, wind_4s, power_4s

def GetData_WSWD(df_wind=None, df_AGC_signal=None, dt=4.0):   
    '''
    
    Wind speed: Resolution is 1min
    AGC: Resolution is 4s
    

    '''
    if df_AGC_signal is None:
        # AGC from Elina
        if running_on_eagle:
            df_AGC_signal = pd.read_csv(
                PKGROOT + "/data/floris_datasets/AGC_simple_regonly_for_training.csv"
            )
        else:
            df_AGC_signal = pd.read_csv(
                "./a2e2g/modules/control/datasets/AGC_simple_regonly_for_training.csv"
            )

    if df_wind is None:
        # Speeds from Andrew
        # CHANGE DATA TO "wind_obs_20190801-20200727_v2.csv"
        if running_on_eagle:
            df_wind = pd.read_csv(
                PKGROOT + "/data/floris_datasets/wind_obs_20190801-20200727_v2.csv"
            )
        else:
            df_wind = pd.read_csv(
                "./a2e2g/modules/control/datasets/wind_obs_20190801-20200727_v2.csv"
            )
        df_wind.rename(columns={'Unnamed: 0':'datetime'}, inplace=True)
        df_wind = df_wind[(df_wind.datetime >= '2019-08-01') & 
                          (df_wind.datetime < '2019-09-01')] # Get only August
        # fills NaNs
        df_wind.interpolate(method='polynomial', order=2, inplace=True)
    
    # Data 
    num_days = round(len(df_wind)/24/60)
    num_sec_day = 86400
    interval_agc = 4
    interval_wind = 60

    df_AGC_signal = df_AGC_signal.iloc[:num_days*num_sec_day//interval_agc]
    df_wind = df_wind.iloc[:num_days*num_sec_day//interval_wind]
    
    # Construct interpolation functions. Use X, Y wind components to avoid 
    # wrapping issues
    time_pow_4s = np.arange(0, num_days*num_sec_day, interval_agc)
    power_4s = df_AGC_signal['Basepoint signal'].to_numpy()
    power_f = interpolate.interp1d(time_pow_4s, power_4s, kind='linear', 
                                   fill_value='extrapolate')

    time_1min = np.arange(0, num_days*num_sec_day, interval_wind)
    ws_U_1min = df_wind['75m_U'].to_numpy()
    ws_U_f = interpolate.interp1d(time_1min, ws_U_1min, kind='quadratic', 
                                  fill_value='extrapolate')
    ws_V_1min = df_wind['75m_V'].to_numpy()
    ws_V_f = interpolate.interp1d(time_1min, ws_V_1min, kind='quadratic', 
                                  fill_value='extrapolate')
    
    # Sample to desired sampling time
    time_dt = np.arange(0, num_days*num_sec_day, dt)
    power_dt = power_f(time_dt)
    ws_U_dt = ws_U_f(time_dt)
    ws_V_dt = ws_V_f(time_dt)

    # Construct absolute wind speed and wind direction    
    ws_dt = np.linalg.norm(np.column_stack([ws_U_dt, ws_V_dt]), axis=1)
    wd_dt = 90-(np.rad2deg(np.arctan2(ws_V_dt, ws_U_dt))+180)
    wd_dt = np.array([wrap_to_360(wd) for wd in wd_dt])
    
    return time_dt, ws_dt, wd_dt, power_dt

if __name__ == '__main__':
    time_pow_4s, wind_4s, power_4s = GetData()

    fig, ax = plt.subplots()
    ax.plot(time_pow_4s, power_4s, label='AGC')
    ax.plot(time_pow_4s, wind_4s, label='wind speed')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('AGC [MW], WS [m/s]')
    plt.show()
