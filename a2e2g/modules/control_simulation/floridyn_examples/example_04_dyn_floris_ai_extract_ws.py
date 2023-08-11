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

from floridyn_special2 import tools as wfct # Incoming con

test_ai_changes = True # Else, test yaw angle change
# NOTE: Static floris doesn't appear to be correct for yaw angles in this case.

floris_dir = "./example_input_3turb.json"
wind_direction = 270.
if test_ai_changes:
    ai_changes = {
        100:[0.2]+[0.33]*2,
        300:[0.33]+[0.33]*2
    }
else:
    angle_changes = {
        250:[10,0,0], 
        500:[10,10,0], 
        750:[20,10,0], 
        1000:[20,20,0]
    }

# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface(floris_dir)

# Calculate wake
fi.calculate_wake()

# Get horizontal plane at default height (hub-height)
hor_plane = fi.get_hor_plane()

powers = []
ss_powers = []
powers_calc = []
total_time = 700
fi.floris.farm.flow_field.mean_wind_speed = 8
turbine_powers = [ [] for turbine in fi.floris.farm.turbines]
ss_turb_powers = [ [] for turbine in fi.floris.farm.turbines]
turbine_powers_calc = [ [] for turbine in fi.floris.farm.turbines]
turbine_wss = [[] for turbine in fi.floris.farm.turbines]
ss_turbine_wss = [[] for turbine in fi.floris.farm.turbines]

# fi.calculate_wake()

yaw_angles = [0 for turbine in fi.floris.farm.turbines]
ai_values = [0.33 for turbine in fi.floris.farm.turbines]
fi.calculate_wake(axial_induction=ai_values, yaw_angles=yaw_angles)

for sim_time in range(total_time):
    if sim_time % 100 == 0:
        print("Iteration:", sim_time)
    
    if test_ai_changes:
        if sim_time in ai_changes:
            ai_values = ai_changes[sim_time]
    else:
        if sim_time in angle_changes:
            yaw_angles = angle_changes[sim_time]
        #fi.reinitialize_flow_field(wind_speed=10, sim_time=sim_time)
    
    #if test_ai_changes:
    fi.calculate_wake(sim_time=sim_time, yaw_angles=yaw_angles, 
        axial_induction=ai_values)
    powers.append(fi.get_farm_power()/1e6)
    powers_calc.append(0)
    for i,turbine in enumerate(fi.floris.farm.turbines):
        turbine_powers[i].append(turbine.power/1e6)
        turbine_wss[i].append(turbine.average_velocity)
        P_calc = 0.5 * turbine.Cp * turbine.air_density * \
                turbine.average_velocity ** 3 * \
                    (turbine.rotor_radius**2 * 3.14159) / 1e6
        turbine_powers_calc[i].append(P_calc)
        powers_calc[sim_time] += P_calc
    
    # Steady-state version
    fi.calculate_wake(yaw_angles=yaw_angles, axial_induction=ai_values)
    ss_powers.append(fi.get_farm_power()/1e6)
    for i,turbine in enumerate(fi.floris.farm.turbines):
        ss_turb_powers[i].append(turbine.power/1e6)
        ss_turbine_wss[i].append(turbine.average_velocity)
    

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)

# Power of individual turbines

plt.figure()
colors = ['C0', 'C1', 'C2']
for i, (dyn_pow, ss_pow, dyn_pow_calc, color) in \
    enumerate(zip(turbine_powers, ss_turb_powers, turbine_powers_calc, colors)):
    label = "Turbine " + str(i)
    plt.plot(list(range(total_time)), dyn_pow, label=label+", dynamic", 
        color=color)
    plt.plot(list(range(total_time)), dyn_pow_calc, 
        label=label+", dynamic, calculated", color='black', linestyle='dashed')
    plt.plot(list(range(total_time)), ss_pow, label=label+", steady-state", 
        color=color, linestyle='dotted')

plt.ylabel("Turbine power (MW)")
plt.xlabel("Time (s)")
plt.legend()

# Total power

plt.figure()

plt.plot(list(range(total_time)), powers, label="Dynamic", color='C0')
plt.plot(list(range(total_time)), powers_calc, label="Dynamic, calculated", 
    color='black', linestyle='dashed')
plt.plot(list(range(total_time)), ss_powers, label="Steady-State", 
    color='C0', linestyle='dotted')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Farm power (MW)")

# Wind speed at individual turbines

plt.figure()
for i, (turbine_ws, tt_ws, color) in \
    enumerate(zip(turbine_wss, ss_turbine_wss, colors)):
    label = "Turbine " + str(i)
    plt.plot(list(range(total_time)), turbine_ws, label=label+", dynamic", 
        color=color)
    plt.plot(list(range(total_time)), tt_ws, label=label+", steady-state", 
        color=color, linestyle='dotted')
plt.ylabel("Wind speed (m/s)")
plt.xlabel("Time (s)")
plt.legend()

plt.show()
