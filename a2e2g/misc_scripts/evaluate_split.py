import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from floris import tools as wfct

import a2e2g.modules.control.RL.Cluster as cluster
import a2e2g.modules.power_forecast.MCPowerEstimation as mcpe


warp_percents = [70, 80, 90]



df_sims = pd.read_pickle('outputs/split_powers.pkl')
N = len(df_sims)
df_control = pd.read_pickle('outputs/control_acts_2019-11-30.pkl')
df_control = df_control.iloc[:N, :]

fig, ax = plt.subplots(2,1)

# Generate a pseudo power 'forecast'
estimator = mcpe.DeterministicPowerEstimation(
    floris_sweep_file="./a2e2g/data_local/floris_datasets/FLORISfullsweep_2.csv",
    ws_sweep=np.linspace(0,25,60), wd_sweep=np.linspace(0,360,50)
)
        
P_for = estimator.predict_power_hist(
    df_control.ws.to_list(),
    df_control.wd.to_list()
)
P_for = np.array(P_for)/1e6

ax[0].plot(df_control.time, df_control.ws)
ax[1].plot(df_control.time, df_control.P_ref/1e6, color='lightgray', linestyle='dashed', label='reference')
ax[1].plot(df_control.time, df_control.P_est_base/1e6, color='black', label='est')
ax[1].plot(df_control.time, df_sims.P_70/1e6, color='darkgreen', label='WindSE 70%')
ax[1].plot(df_control.time, df_sims.P_80/1e6, color='seagreen', label='WindSE 80%')
ax[1].plot(df_control.time, df_sims.P_90/1e6, color='lightseagreen', label='WindSE 90%')
ax[1].plot(df_control.time, [13.232]*N, color='red', linestyle='dashed', label='5 min. forecast')
ax[1].plot(df_control.time, P_for, color='blue', linestyle='dashed', label='Curr. forecast')

ax[1].legend()
ax[1].set_ylabel('P [MW]')
ax[1].set_xlabel('Time [s]')
ax[0].set_ylabel('WS [m/s]')
# P_ref
# P_est
# P_70, 80, 90

plt.show()

