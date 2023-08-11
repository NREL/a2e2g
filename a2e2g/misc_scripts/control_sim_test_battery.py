# inputs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from a2e2g import a2e2g # module containing wrappers to different components
# modified dynamic version of floris with axial
import a2e2g.modules.control.floris.tools as wfct # modified with axial induction direct access.
import a2e2g.modules.control.RL.RLcontrol as control
import a2e2g.modules.control.RL.Cluster as cluster
import a2e2g.modules.control_simulation.truth_sim_battery as simulate_battery

import matplotlib
import time
matplotlib.use('TKAgg') # Some weird workaround, maybe only for Macs/me

if __name__ == '__main__': # For multiprocessing in a2e2g.py

    data_directory = '/projects/a2e2g/data_share'
    data_directory = './a2e2g/data_local'

    sim = a2e2g.a2e2g(data_directory)


    wind_data_file = './a2e2g/modules/control/datasets/wind_obs_20190801-20200727_v2.csv'
    startTimeSim = "2019-11-30"
    stopTimeSim = "2019-12-01"

    df_wind = pd.read_pickle(data_directory+'/test_data/df_wind_testing_DAbid.p')
    AGC = pd.read_pickle(data_directory+'/test_data/df_AGC_testing_DAbid.p')

    sim_range = [datetime(2019, 9, 28, 4, 30), datetime(2019, 9, 28, 5, 0)]

    ###
    crm = 16
    batt = simulate_battery.SimpleSOC(
            capacity=simulate_battery.BatterySimulator.MWh_to_J(crm*2),
            charge_rate_max=crm*1e6,   ### ^^ 2 hour battery of each size.
            discharge_rate_max=crm*1e6,
            charging_efficiency=0.92,
            discharging_efficiency=0.92,
            storage_efficiency=(1-1e-5),
            initial_SOC=simulate_battery.BatterySimulator.MWh_to_J(crm*2)*0.999,
            dt=1.0
        )

    ###

    Proportional_only = False
    if Proportional_only:
        controllers=['base', 'P', 'P_b'] # ['base', 'P']
        use_battery=[False, False, True]
    else:
        controllers=['P_b']
        use_battery=[True]
        battery_simulators=[batt]
    
    t_start = time.perf_counter()
    df_control = sim.simulate_operation(
        control_cases=controllers,
        use_battery=use_battery,
        df_wind=df_wind,
        df_AGC_signal=AGC,
        closed_loop_simulator='FLORIDyn',
        sim_range=sim_range,
        dt=1,
        nondefault_batteries=battery_simulators,
        parallelize_over_cases=False
    )
    t_stop = time.perf_counter()
    print('Simulating {0} cases took {1:.2f} seconds.'.format(len(controllers),
        t_stop-t_start))

    fig, ax = plt.subplots(5, 1, sharex=True)
    ax[0].plot(df_control.time, df_control.ws)
    ax[1].plot(df_control.time, df_control.wd)
    ax[3].plot(df_control.time, df_control.P_ref/1e6, color='black', linestyle='dashed', label='AGC')
    control_colors=['C0', 'C1', 'C2']
    for c, cc, ub in zip(controllers, control_colors, use_battery):
        control_actions = np.stack(df_control['ai_cmd_'+c].values)
        for t in range(np.shape(control_actions)[1]):
            ax[2].plot(df_control.time, control_actions[:,t], color=cc, alpha=0.1)
        ax[3].plot(df_control.time, df_control['P_act_'+c]/1e6, color=cc, label=c)
        if ub:
            ax[3].plot(df_control.time, df_control['P_farm_'+c]/1e6, color=cc, 
                linestyle='dashed', label=c+'_farm')
            ax[4].plot(df_control.time, df_control['E_batt_'+c]/(1e6*60*60), 
                color=cc, label=c)
    ax[3].legend()

    ax[0].set_ylabel('Wind speed [m/s]')
    ax[1].set_ylabel('Wind dir. [deg]')
    ax[2].set_ylabel('Ax. ind. [-]')
    ax[3].set_ylabel('Power [MW]')
    ax[4].set_xlabel('Time [s]')
    ax[4].set_ylabel('Battery Energy [MWh]')

    # Bit hacky, but let's also plot the SOC limits
    ax[4].plot(df_control.time, [crm*2]*len(df_control), color='black',
        linestyle='dashed')
    ax[4].plot(df_control.time, [crm*2*0.95]*len(df_control), color='black',
        linestyle='dashed')

    plt.show()

# print(self.problem.u_k(539.195, 1525.514, 80.0))