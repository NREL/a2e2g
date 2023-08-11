# inputs
from datetime import date, datetime
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
data_directory = './a2e2g/data_local'

sim = a2e2g.a2e2g(data_directory)

df_wind = pd.read_pickle(data_directory+'/test_data/df_wind_testing.p')

AGC = pd.read_pickle(data_directory+'/test_data/df_AGC_testing.p')

datetime_wind = pd.date_range("2020-01-01", "2020-01-01 23:59:56",freq='1min')
np.random.seed(0)
U_base = 8*np.ones(len(datetime_wind))
U_rand = np.random.normal(loc=0.0, scale=0.0, size=len(datetime_wind))
V_base = 1*np.ones(len(datetime_wind))
V_rand = np.random.normal(loc=0.0, scale=0.0, size=len(datetime_wind))
df_wind = pd.DataFrame({
    'datetime': datetime_wind,
    '75m_U': U_base + U_rand,
    '75m_V': V_base + V_rand
})
agc_times = pd.date_range("2020-01-01", "2020-01-01 23:59:56", freq='4s')
bp_signal = [19]*100 + \
            [20]*100 + \
            [22]*100 + \
            [20]*50 + \
            [19]*50 + \
            [19]*(len(agc_times) - 400)
AGC = pd.DataFrame({
    'time': agc_times,
    'Basepoint signal': bp_signal
})
AGC = pd.read_pickle(data_directory+'/test_data/df_AGC_testing.p')

#n_per_day = round(60*60*24/4) # 4s intervals
#AGC = AGC.iloc[-n_per_day:]

PIonly = False
if PIonly:
    controllers=['PI']
else:
    controllers=['base', 'PI']
    controllers=['base', 'wake_steering']
df_control = sim.simulate_operation(
    control_cases=controllers,
    df_wind=df_wind, 
    df_AGC_signal=AGC,
    closed_loop_simulator='FLORIDyn',
    sim_range=[4275, 4500],#6525],
    dt=1.0,
    parallelize_over_cases=False
)

fig, ax = plt.subplots(5, 1)
ax[0].plot(df_control.time, df_control.ws)
ax[1].plot(df_control.time, df_control.wd)
ax[2].plot(df_control.time, df_control.P_ref/1e6, color='red', label='AGC')
if PIonly:
    control_colors=['green']
else:
    control_colors=['blue', 'green']
for c, cc in zip(controllers, control_colors):
    ax[2].plot(df_control.time, df_control['P_act_'+c]/1e6, color=cc, label=c)
    ai_controls = np.stack(df_control['ai_cmd_'+c].values)
    yaw_controls = np.stack(df_control['yaw_cmd_'+c].values)
    for t in range(np.shape(ai_controls)[1]):
        ax[3].plot(df_control.time, ai_controls[:,t], color=cc, alpha=0.1)
        ax[4].plot(df_control.time, yaw_controls[:,t], color=cc, alpha=0.1)
ax[2].legend()

ax[0].set_ylabel('Wind speed [m/s]')
ax[1].set_ylabel('Wind dir. [deg]')
ax[2].set_ylabel('Power [MW]')
ax[3].set_ylabel('Ax. ind. [-]')
ax[4].set_ylabel('Yaw mis. [deg]')
ax[4].set_xlabel('Time [s]')

print(df_control.columns)
plt.show()

# print(self.problem.u_k(539.195, 1525.514, 80.0))