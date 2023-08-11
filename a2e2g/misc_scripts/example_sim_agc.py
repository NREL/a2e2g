# main driver for all the a2e2g parts

from site import USER_BASE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# import forecast # Andrew's forecasting code
# import floris_estimation # wrapper to call FLORIS code
# import day_ahead_bidding # wrapper to call Elina's code

# import intermediate_bidding # wrapper to call Elina's code

# import floris_short_term # wrapper to call FLORIS code
# import real_time_AGC_signal # wrapper to call Elina's code
# import RL_controller # wrapper to RL controller
# import WindSE
# import compute_value # wrapper to value code

# import floris.tools as wfct # FLORIS

from a2e2g import a2e2g # module containing wrappers to different components

import a2e2g.modules.control_simulation.truth_sim_battery as simulate_battery
import a2e2g.modules.market.market as mkrt

import matplotlib.pyplot as plt

if __name__ == '__main__': # For multiprocessing in a2e2g.py

    ###########################
    # Day Ahead Participation #
    ###########################

    running_on_eagle = False
    use_pregenerated_data = False

    # inputs
    wind_speed = []
    wind_direction = []
    TI = []

    save_output = True

    if running_on_eagle:
        data_directory = '/projects/a2e2g/data_share'
    else:
        data_directory = './a2e2g/data_local'

    sim = a2e2g.a2e2g(data_directory)

    # Configure the forecast parameters
    utc_minus = 6
    tz_current = 'US/Central'
    tz_adjust = 18
    if running_on_eagle:
        wind_data_file = data_directory + '/forecast/wind_obs_20190801-20200727_v2.csv'
        hrrr_files = data_directory + '/forecast/hrrr/*.csv*'
    else:
        wind_data_file = './a2e2g/modules/control/datasets/wind_obs_20190801-20200727_v2.csv'
        hrrr_files = data_directory + '/forecast/hrrr/*.csv*'

    # Configure the market parameters
    startTimeSim = "2019-09-28"
    stopTimeSim = "2019-09-28"
    market_name = "ERCOT"
    # all options for ERCOT ["HB_SOUTH","HB_NORTH","HB_WEST", "HB_BUSAVG"]
    bus="HB_BUSAVG"

    # Generate a figure to start plotting things
    Nplots = 7
    fig, ax = plt.subplots(Nplots,1,sharex=True)

    # Forecast day ahead parameters
    forecast_outputs = sim.forecast(
        utc_minus, tz_current, tz_adjust, wind_data_file, hrrr_files, day=startTimeSim, integration=use_pregenerated_data
    )

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

    # print(forecast_outputs[1])
    # lkj

    # Estimate wind plant power production
    #floris_outputs = sim.corrected_floris_estimation(forecast_outputs, integration=False)
    floris_outputs = sim.floris_estimation(forecast_outputs)
    floris_outputs.PWR_MEAN = floris_outputs.PWR_MEAN/1e6
    floris_outputs.PWR_STD = floris_outputs.PWR_STD/1e6

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

    short_term_power_estimate = sim.floris_deterministic(df_st_forecast)
    short_term_power_estimate.PWR_MEAN = short_term_power_estimate.PWR_MEAN/1e6
    short_term_power_estimate.PWR_STD = short_term_power_estimate.PWR_STD/1e6

    print('here')
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
    print(RTbid)

    df_RT_result = market.real_time_market_simulation(df_RT_result, RTbid, RTprices)

    print(df_RT_result)

    AGC = market.create_system_regulation_signal(create_AGC=(not use_pregenerated_data))

    print(AGC)

    AGC = market.create_wind_plant_regulation_signal(AGC, df_RT_result)

    print(AGC)


    df_wind = sim.load_test_winds(wind_data_file, daterange=(startTimeSim,stopTimeSim))

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

    print(df_wind)

    # Generate the batteries
    crm = 4
    battery_simulator = simulate_battery.SimpleSOC(
            capacity=simulate_battery.BatterySimulator.MWh_to_J(crm*4),
            charge_rate_max=crm*1e6,   ### ^^ 4 hour battery
            discharge_rate_max=crm*1e6,
            charging_efficiency=0.92,
            discharging_efficiency=0.92,
            storage_efficiency=(1-1.18e-8),
            initial_SOC=simulate_battery.BatterySimulator.MWh_to_J(crm*4)*0.5, 
            dt=1.0
        )

    all_batteries = [None, battery_simulator, None, battery_simulator]

    print("Beginning closed-loop simulation.")

    # Generates controls, simulates over a short period.
    df_sim = sim.simulate_operation(
        control_cases=['base', 'base', 'P', 'P'],
        use_battery=[False, True, False, True],
        df_wind=df_wind,
        df_AGC_signal=AGC,
        closed_loop_simulator='FLORIDyn',
        sim_range=[datetime(2019, 9, 28, 4, 45), datetime(2019, 9, 28, 6, 00)],
        dt=1.0,
        nondefault_batteries=all_batteries,
        name_extensions=['', '_batt', '', '_batt'],
        parallelize_over_cases=True
    )

    if save_output:
        df_sim.to_pickle('df_control_sim_agc.p')

    print(df_sim)
    ax[6].plot(df_sim.time, df_sim['P_act_base']/1e6, 
        color='blue', label='base')
    ax[6].plot(df_sim.time, df_sim['P_act_P']/1e6, 
        color='green', label='PI')

    plt.show()