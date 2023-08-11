# main driver for all the a2e2g parts

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from a2e2g import a2e2g # module containing wrappers to different components

import a2e2g.modules.market.market as mkrt

# inputs
data_directory = "data"
wind_plant = 'staggered_50MW'


# Establish main simulation class to call modules
sim = a2e2g.a2e2g(data_directory, wind_plant)

# Configure the forecast parameters
utc_minus = 6
tz_current = 'US/Central'
tz_adjust = 18

# Data paths (data not stored in github repository)
wind_data_file = 'a2e2g/'+data_directory+'/measured/wind_obs_20190801-20200727_v2.csv'
hrrr_files = 'a2e2g/'+data_directory+'/forecast/hrrr/*.csv*'

# Configure the market parameters
startTimeSim = "2019-09-28"
stopTimeSim = "2019-09-28"
market_name = "ERCOT"
# all options for ERCOT ["HB_SOUTH","HB_NORTH","HB_WEST", "HB_BUSAVG"]
bus="HB_BUSAVG"

# Generate a figure to start plotting things
Nplots = 7
fig, ax = plt.subplots(Nplots,1,sharex=True)

###########################
# Day Ahead Participation #
###########################

# Forecast day ahead parameters
forecast_outputs = sim.forecast(
    utc_minus, tz_current, tz_adjust, wind_data_file, hrrr_files, day=startTimeSim
)

# Generate plot of forecast
ax[0].plot(forecast_outputs[0]['Analog Mean'], color='black')
ax[0].fill_between(forecast_outputs[0].index,
    forecast_outputs[0]['Analog Mean']+forecast_outputs[0]['Analog Stde'],
    forecast_outputs[0]['Analog Mean']-forecast_outputs[0]['Analog Stde'], 
    color='black', alpha=0.2)
ax[0].set_ylabel('DA WS [m/s]')
ax[1].plot(forecast_outputs[1]['Analog Mean'], color='black')
ax[1].fill_between(forecast_outputs[0].index,
    forecast_outputs[1]['Analog Mean']+forecast_outputs[1]['Analog Stde'],
    forecast_outputs[1]['Analog Mean']-forecast_outputs[1]['Analog Stde'], 
    color='black', alpha=0.2)
ax[1].set_ylabel('DA WD [deg]')

# Estimate wind plant power production
floris_outputs = sim.floris_estimation(forecast_outputs, scale_to_MW=True)

# Plot power forecast
ax[2].plot(floris_outputs.Time, (floris_outputs.PWR_MEAN), color='black')
ax[2].fill_between(floris_outputs.Time,
    (floris_outputs.PWR_MEAN+floris_outputs.PWR_STD),
    (floris_outputs.PWR_MEAN-floris_outputs.PWR_STD),
    color='black', alpha=0.2)
ax[2].set_ylabel('DA pow [MW]')

# Initialize market object
market = mkrt.Market(startTimeSim, stopTimeSim, market_name, bus, data_directory=data_directory)
dfrt2, dfrt2AS, day_ahead_prices, RTprices = sim.load_price_data(market)

# Determine day ahead market bid to make
day_ahead_bid = sim.day_ahead_bidding(market, dfrt2, dfrt2AS, floris_outputs)

# #################################
# # Intermediate Day Ahead Market #
# #################################

# use day ahead bid to determine intermediate bid
intermediate_bid = sim.intermediate_bidding(market, day_ahead_bid, day_ahead_prices)

# ####################
# # Real-time Market #
# ####################

df_st_forecast = sim.short_term_persistence(
    wind_data_file, daterange=(startTimeSim,stopTimeSim)
)

short_term_power_estimate = sim.floris_deterministic(df_st_forecast, scale_to_MW=True)

# Add more info to plots
ax[3].plot(forecast_outputs[0]['Analog Mean'], color='black')
ax[3].fill_between(forecast_outputs[0].index,
    forecast_outputs[0]['Analog Mean']+forecast_outputs[0]['Analog Stde'],
    forecast_outputs[0]['Analog Mean']-forecast_outputs[0]['Analog Stde'], 
    color='black', alpha=0.2)
ax[3].set_ylabel('ST WS [m/s]')
ax[4].plot(forecast_outputs[1]['Analog Mean'], color='black')
ax[4].fill_between(forecast_outputs[0].index,
    forecast_outputs[1]['Analog Mean']+forecast_outputs[1]['Analog Stde'],
    forecast_outputs[1]['Analog Mean']-forecast_outputs[1]['Analog Stde'], 
    color='black', alpha=0.2)
ax[4].set_ylabel('ST WD [deg]')
ax[5].plot(floris_outputs.Time, (floris_outputs.PWR_MEAN), color='black')
ax[5].fill_between(floris_outputs.Time,
    (floris_outputs.PWR_MEAN+floris_outputs.PWR_STD),
    (floris_outputs.PWR_MEAN-floris_outputs.PWR_STD),
    color='black', alpha=0.2)
ax[5].set_ylabel('ST pow [MW]')
ax[3].plot(df_st_forecast.Time, df_st_forecast.WS, color='red')
ax[4].plot(df_st_forecast.Time, df_st_forecast.WD, color='red')
ax[5].plot(short_term_power_estimate.Time, (short_term_power_estimate.PWR_MEAN), color='red')


RTbid, df_RT_result = sim.real_time_AGC_signal(
    market, intermediate_bid, short_term_power_estimate
)

df_RT_result = market.real_time_market_simulation(df_RT_result, RTbid, RTprices)

# #############################
# # Real-time plant operation #
# #############################

# Generate AGC signals for system and plant
AGC = market.create_system_regulation_signal(create_AGC=True)
AGC = market.create_wind_plant_regulation_signal(AGC, df_RT_result)

# Plot
agc_times = pd.date_range(
    df_st_forecast.Time.iloc[0], 
    df_st_forecast.Time.iloc[-1]+pd.Timedelta(5, 'm')-pd.Timedelta(4, 's'), 
    freq='4s'
)
AGC['time'] = agc_times
ax[6].plot(AGC.time, AGC['Basepoint signal'], label='AGC')
ax[6].set_ylabel('Power [MW]')
ax[6].plot(short_term_power_estimate.Time, (short_term_power_estimate.PWR_MEAN), 
    label='ST forecast', color='red')

# Load a test simultion wind case
df_wind = sim.load_test_winds(wind_data_file, daterange=(startTimeSim,stopTimeSim))

# Generate controls, simulate over a short period with power maximizing 
# control ('base') and error-proportional plant-level active power control 
# ('P').

df_sim = sim.simulate_operation(control_cases=['base', 'P'],
    df_wind=df_wind, df_AGC_signal=AGC, closed_loop_simulator='FLORIDyn',
    sim_range=[datetime(2019, 9, 28, 5, 45), datetime(2019, 9, 28, 6, 15)],
    dt=1.0)

# Plot controller responses
ax[6].plot(df_sim.time, df_sim['P_act_base']/1e6, 
    color='blue', label='base')
ax[6].plot(df_sim.time, df_sim['P_act_P']/1e6, 
    color='green', label='P')

plt.show()