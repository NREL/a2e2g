# Script for testing various battery sizes with the proportional 
# wind plant controller to determine performance in following the 
# day ahead bid.


# inputs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd
from datetime import datetime

from a2e2g import a2e2g # module containing wrappers to different components
# modified dynamic version of floris with axial
import a2e2g.modules.control_simulation.truth_sim_battery as simulate_battery
import a2e2g.modules.market.market_da_tracking as mkrt

import matplotlib
import time
matplotlib.use('TKAgg') # Some weird workaround, maybe only for Macs/me

if __name__ == '__main__': # For multiprocessing in a2e2g.py

    save_output = True

    data_directory = '/projects/a2e2g/data_share'
    data_directory = './a2e2g/data_local'
    wind_plant = 'staggered_50MW'
    wind_data_file = './a2e2g/modules/control/datasets/wind_obs_20190801-20200727_v2.csv'
    startTimeSim = "2019-09-28"
    stopTimeSim = "2019-09-28"

    sim = a2e2g.a2e2g(data_directory, wind_plant)

    print("Loading and handling wind information...")
    df_wind = sim.load_test_winds(wind_data_file, daterange=(startTimeSim,stopTimeSim))
    
    # Construct 'forecasted' winds
    df_wind['WS_abs'] = \
        np.linalg.norm(df_wind[['75m_U', '75m_V']].to_numpy(), axis=1)
    df_wind.set_index('datetime', inplace=True)
    ws_mean = df_wind['WS_abs'].resample('5T').mean()
    ws_std = df_wind['WS_abs'].resample('5T').std()
    wd_mean = (df_wind['75m_WD']*np.pi/180).resample('5T').\
        apply(circmean)*180/np.pi
    wd_std = (df_wind['75m_WD']*np.pi/180).resample('5T').\
        apply(circstd)*180/np.pi

    forecast_outputs = [pd.DataFrame(index=ws_mean.index,
        data={'Analog Mean':ws_mean, 'Analog Stde':ws_std}),
        pd.DataFrame(index=wd_mean.index,
        data={'Analog Mean':wd_mean, 'Analog Stde':wd_std}),
    ]
    print("Wind information loaded.")

    print("Generating power forecasts...")
    floris_outputs = sim.floris_estimation(forecast_outputs)
    floris_outputs.PWR_MEAN = floris_outputs.PWR_MEAN/1e6
    floris_outputs.PWR_STD = floris_outputs.PWR_STD/1e6
    print("Power forecasts generated.")

    # Initialize market object
    print("Interacting with market...")
    market = mkrt.Market(startTimeSim, stopTimeSim, "ERCOT", "HB_BUSAVG", data_directory=data_directory)
    dfrt2, dfrt2AS, day_ahead_prices, RTprices = sim.load_price_data(market)
    # Determine day ahead market bid to make
    day_ahead_bid = sim.day_ahead_bidding(market, dfrt2, dfrt2AS, floris_outputs)
    dab_array = np.array([hour_bid[0] for hour_bid in day_ahead_bid.to_numpy()[0,:]])

    AGC = pd.DataFrame({'time': pd.date_range(
        df_wind.index[0], 
        df_wind.index[-1]+pd.Timedelta(1, 'm')-pd.Timedelta(4, 's'), 
        freq='4s'),
        'Basepoint signal':np.repeat(dab_array, 60*15) # Generate 4s intervals
    })
    print("Bids made and reference signal created.")
     
    # Take a look at the day
    fig, ax = plt.subplots(1,1)
    floris_outputs.plot('Time', 'PWR_MEAN', label='forecast', ax=ax)
    temp_df = floris_outputs.set_index('Time').resample('1H').mean().reset_index()
    AGC.plot('time', 'Basepoint signal', label='Power reference', color='black', ax=ax)
    ax.grid()
    # plt.show()
    #import ipdb; ipdb.set_trace()

    sim_range = [datetime(2019, 9, 28, 6, 0), datetime(2019, 9, 28, 18, 0)]
    isoc_fraction = 0.5
    dt_sim = 1.0

    battery_P_caps_MW = [2, 4, 8, 16] 
    controllers=['base']*(len(battery_P_caps_MW)+1) + \
        ['P']*(len(battery_P_caps_MW)+1)
    use_battery=([False] + [True]*len(battery_P_caps_MW))*2
    name_extensions = (['']+['_2MW', '_4MW', '_8MW', '_16MW'])*2
    
    # Construct battery simulators
    battery_simulators = ([None] + [simulate_battery.SimpleSOC(
            capacity=simulate_battery.BatterySimulator.MWh_to_J(crm*4),
            charge_rate_max=crm*1e6,   ### ^^ 4 hour battery of each size.
            discharge_rate_max=crm*1e6,
            charging_efficiency=0.92,
            discharging_efficiency=0.92,
            storage_efficiency=(1-1.18e-8),
            initial_SOC=simulate_battery.BatterySimulator.MWh_to_J(crm*4)*\
                isoc_fraction, 
            dt=dt_sim
        ) for crm in battery_P_caps_MW])*2
    
    t_start = time.perf_counter() # For simulation timing
    
    df_control = sim.simulate_operation(
        control_cases=controllers,
        use_battery=use_battery,
        df_wind=df_wind,
        df_AGC_signal=AGC,
        closed_loop_simulator='FLORIDyn',
        sim_range=sim_range,
        dt=dt_sim,
        nondefault_batteries=battery_simulators,
        name_extensions=name_extensions,
        parallelize_over_cases=True
    )
    t_stop = time.perf_counter()
    print('Simulating {0} cases took {1:.2f} seconds.'.format(len(controllers),
        t_stop-t_start))
    
    if save_output:
        df_control.to_pickle('df_control_sim_6to6.p')

    fig, ax = plt.subplots(5, 1, sharex=True)
    ax[0].plot(df_control.time, df_control.ws)
    ax[1].plot(df_control.time, df_control.wd)
    ax[3].plot(df_control.time, df_control.P_ref/1e6, color='black', linestyle='dashed', label='AGC')
    control_colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for c, ce, cc, ub in zip(controllers, 
                             name_extensions, 
                             control_colors, 
                             use_battery):
        #control_actions = np.stack(df_control['P_cmd_'+c+ce].values)
        #import ipdb; ipdb.set_trace()
        ai_positions = np.stack(df_control['ai_'+c].values)
        for t in range(np.shape(ai_positions)[1]):
            ax[2].plot(df_control.time, ai_positions[:,t], color=cc, alpha=0.1)
            #ax[3].plot(df_control.time, control_actions[:,t], color=cc, alpha=0.1)
        ax[3].plot(df_control.time, df_control['P_act_'+c+ce]/1e6, color=cc, label=c)
        if ub:
            ax[3].plot(df_control.time, df_control['P_farm_'+c+ce]/1e6, color=cc, 
                linestyle='dashed', label=c+'_farm')
            ax[4].plot(df_control.time, df_control['E_batt_'+c+ce]/1e6, 
                color=cc, label=c)
    ax[3].legend()

    ax[0].set_ylabel('Wind speed [m/s]')
    ax[1].set_ylabel('Wind dir. [deg]')
    ax[2].set_ylabel('Ax. ind. [-]')
    ax[3].set_ylabel('Power [MW]')
    ax[4].set_xlabel('Time [s]')
    ax[4].set_ylabel('Battery Energy [MJ]')

    # Bit hacky, but let's also plot the SOC limits
    ax[4].plot(df_control.time, [72000.0]*len(df_control), color='black',
        linestyle='dashed')
    ax[4].plot(df_control.time, [0]*len(df_control), color='black', 
    linestyle='dashed')

    plt.show()