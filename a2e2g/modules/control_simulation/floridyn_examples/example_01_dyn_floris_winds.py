# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import numpy as np
from floridyn_special2 import tools as wfct # Incoming con
#import ffmpeg

# **************************************** Parameters **************************************** #

# total simulation time
total_time = 600
dt = 1.0 # DOESN'T WORK WELL WITH OTHER DT VALUES

# Test varying wind speed and direction? Or test one at a time? 
# Step change and random variation will be added.
test_varying_ws = True
test_varying_wd = True

# **************************************** Initialization **************************************** #
# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
floris_dir = "./example_input_3turb.json"
start_wind_dir = 270. 

# Initialize
fi = wfct.floris_interface.FlorisInterface(floris_dir)
fi.reinitialize_flow_field(wind_speed=8, wind_direction=start_wind_dir) 
fi.calculate_wake() 

# Get horizontal plane at default height (hub-height)
# NOTE: this is currently commented out, wind conditions 
# at sim_time 0 below will be assumed to be initial conditions
#hor_plane = fi.get_hor_plane()

# lists that will be needed for visualizations
yaw_angles = [0 for turbine in fi.floris.farm.turbines]
ais = [0.33 for turbine in fi.floris.farm.turbines]
powers = []
true_powers = []
turbine_velocities = []
hor_planes = []
wind_vectors = []
iterations = []
turbine_wind_speeds = [ [] for turbine in fi.floris.farm.turbines]
turbine_powers = [ [] for turbine in fi.floris.farm.turbines]

# **************************************** Simulation **************************************** #

TI = 5

mean = 8
dev = (TI/100)*mean

wind_speed_profile_high_ti = {}

np.random.seed(0)

for i in range(total_time):
    wind_speed_profile_high_ti[i] = np.random.normal(loc=mean, scale=dev)#np.random.uniform(low=8, high=8.3)
#wind_speed_profile_high_ti[total_time] = np.nan

fi.floris.farm.flow_field.mean_wind_speed = 8
if test_varying_ws:
    # Assumes dt = 1... not very clean
    wind_speeds_base = [8]*100 + [10]*100 + [10]*(total_time-200)
else:
    wind_speeds_base = [8]*total_time

if test_varying_wd:
    wind_dirs_base = [start_wind_dir]*100 + [start_wind_dir]*100 + [start_wind_dir-20]*(total_time-200)
else:
    wind_dirs_base = [start_wind_dir]*total_time
    
for sim_time, ws, wd in zip(np.arange(0, total_time*dt, dt), wind_speeds_base, wind_dirs_base):
    iterations.append(sim_time)
    if sim_time % 100 == 0:
        print("Iteration:", sim_time)

    if sim_time >= 1:
        ws_dev = np.random.normal(scale=0.5) if test_varying_ws else 0.
        wd_dev = np.random.normal(scale=5) if test_varying_wd else 0.
        fi.reinitialize_flow_field(wind_speed=ws+ws_dev, wind_direction=wd+wd_dev, sim_time=sim_time)
    
    # calculate dynamic wake computationally
    fi.calculate_wake(sim_time=sim_time, yaw_angles=yaw_angles, axial_induction=ais)
    
    
    # Is this the part I need? no
    for i,turbine in enumerate(fi.floris.farm.turbines):
       turbine_wind_speeds[i].append(fi.floris.farm.wind_map.turbine_wind_speed[i])


    for i,turbine in enumerate(fi.floris.farm.turbines):
        turbine_powers[i].append(turbine.power/1e6)

    # NOTE: at this point, can also use measure other quantities like average velocity at a turbine, etc.
    powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)
    
    # calculate steady-state wake computationally
    fi.calculate_wake(yaw_angles=yaw_angles, axial_induction=ais)

    # NOTE: at this point, can also use measure other quantities like average velocity at a turbine, etc.
    true_powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)

# **************************************** Plots/Animations **************************************** #
# Plot and show
plt.figure()

for i,turbine_power in enumerate(turbine_powers):
    label = "Turbine " + str(i)
    plt.plot(list(np.arange(0, total_time*dt, dt)), turbine_power, label=label)
plt.ylabel("Power (MW)")
plt.xlabel("Time (s)")
plt.legend()

plt.figure()

for i,turbine_wind_speed in enumerate(turbine_wind_speeds):
    label = "Turbine " + str(i)
    plt.plot(list(np.arange(0, total_time*dt, dt)), turbine_wind_speed, label=label)
plt.ylabel("Wind Speed (m/s)")
plt.xlabel("Time (s)")
plt.legend()

plt.figure()

plt.plot(list(np.arange(0, total_time*dt, dt)), powers, label="Dynamic")
plt.plot(list(np.arange(0, total_time*dt, dt)), true_powers, label="Steady-State")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Power (MW)")

plt.show()
