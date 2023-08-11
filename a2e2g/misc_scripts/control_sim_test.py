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

    df_wind = pd.read_pickle(data_directory+'/test_data/df_wind_testing.p')
    AGC = pd.read_pickle(data_directory+'/test_data/df_AGC_testing_DAbid.p')

    sim_range = [datetime(2019, 9, 28, 5, 45), datetime(2019, 9, 28, 5, 50)]

    controllers=['base', 'P'] #, 'P']#, 'P_ws']#, 'P_ws']#, 'PI']
    if 'wake_steering' in controllers and False: # CHANGE TO TRUE
        # Replace real AGC signal with a wake steering active/inactive signal
        interval_mins = 1
        repetitions = len(AGC) // (15*interval_mins*2)
        updown = np.tile(
            np.concatenate((np.zeros(interval_mins*15, dtype=bool),
                            -1*np.ones(interval_mins*15, dtype=bool))),
            repetitions
        )
        AGC["Basepoint signal"] = updown

    t_start = time.perf_counter()
    df_control = sim.simulate_operation(
        control_cases=controllers,
        df_wind=df_wind, 
        df_AGC_signal=AGC,
        closed_loop_simulator='FLORIDyn',
        sim_range=sim_range,
        dt=1.0,
        parallelize_over_cases=True #
    )
    t_stop = time.perf_counter()
    print('Simulating {0} cases took {1:.2f} seconds.'.format(len(controllers),
        t_stop-t_start))

    col_cmd = 'black'
    fig, ax = plt.subplots(6, 1, sharex=True)
    ax[0].plot(df_control.time, df_control.ws)
    ax[1].plot(df_control.time, df_control.wd)
    ax[4].plot(df_control.time, df_control.P_ref/1e6, color='red', label='AGC')
    control_colors=['blue', 'orange', 'green']
    for c, cc in zip(controllers, control_colors):
        T_power_controls = np.stack(df_control['P_cmd_'+c].values)
        ai_positions = np.stack(df_control['ai_'+c].values)
        yaw_controls = np.stack(df_control['yaw_cmd_'+c].values)
        yaw_positions = np.stack(df_control['yaw_abs_'+c].values)
        T_powers = np.stack(df_control['P_turbs_'+c].values)
        for t in range(np.shape(T_power_controls)[1]):
            ax[1].plot(df_control.time, yaw_positions[:,t], color=cc, alpha=0.1)
            ax[5].plot(df_control.time, T_power_controls[:,t], color=col_cmd, alpha=0.1)
            ax[2].plot(df_control.time, ai_positions[:,t], color=cc, alpha=0.1)
            ax[3].plot(df_control.time, yaw_controls[:,t], color=col_cmd, alpha=0.1)
            ax[5].plot(df_control.time, T_powers[:,t], color=cc, alpha=0.1)
        ax[4].plot(df_control.time, df_control['P_act_'+c]/1e6, color=cc, label=c)
    ax[4].legend()

    ax[0].set_ylabel('Wind speed [m/s]')
    ax[1].set_ylabel('Wind dir. [deg]')
    ax[2].set_ylabel('Ax. ind. [-]')
    ax[3].set_ylabel('Yaw pos. [deg]')
    ax[4].set_ylabel('Farm power [MW]')
    ax[5].set_ylabel('Turb power [MW]')
    ax[4].set_xlabel('Time [s]')

    plt.show() # comment out when profiling

#    df_control.to_pickle('df_control_test_2.p')

    # print(self.problem.u_k(539.195, 1525.514, 80.0))