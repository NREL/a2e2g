# inputs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from a2e2g import a2e2g # module containing wrappers to different components
# modified dynamic version of floris with axial
import a2e2g.modules.control.floris.tools as wfct # modified with axial induction direct access.
import a2e2g.modules.control.RL.RLcontrol as control
import a2e2g.modules.control.RL.Cluster as cluster

import matplotlib
import time
matplotlib.use('TKAgg') # Some weird workaround, maybe only for Macs/me

data_directory = '/projects/a2e2g/data_share'

sim = a2e2g.a2e2g(data_directory)


wind_data_file = './a2e2g/modules/control/datasets/wind_obs_20190801-20200727_v2.csv'
startTimeSim = "2019-11-30"
stopTimeSim = "2019-12-01"

df_wind = sim.load_test_winds(wind_data_file, daterange=(startTimeSim,stopTimeSim))

AGC = pd.read_csv('./a2e2g/modules/control/datasets/AGC_simple_regonly_for_training.csv')

n_per_day = round(60*60*24/4) # 4s intervals
AGC = AGC.iloc[-n_per_day:]

df_control = sim.simulate_operation(
    control_cases=['PI'], 
    df_wind=df_wind, 
    df_AGC_signal=AGC,
    closed_loop_simulator=None,
    sim_range=[4275, 6525]
)

print(df_control)

agc_times = pd.date_range(
    "2019-11-30", 
    "2019-11-30 23:59:56",
    freq='4s'
)

fig, ax = plt.subplots(1, 1)
ax.plot(df_control.time, df_control.P_ref/1e6, color='red', label='AGC')
ax.plot(df_control.time, df_control.P_est_PI/1e6, linestyle='dashed', 
    color='black', label='control')
ax.legend()

plt.show()

# print(self.problem.u_k(539.195, 1525.514, 80.0))