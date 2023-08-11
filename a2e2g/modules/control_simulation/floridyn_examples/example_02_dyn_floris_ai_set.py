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
        250:[0.2]+[0.33]*2
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
true_powers = []
total_time = 700
fi.floris.farm.flow_field.mean_wind_speed = 8
turbine_powers = [ [] for turbine in fi.floris.farm.turbines]
true_turb_powers = [ [] for turbine in fi.floris.farm.turbines]
# turb_0_yaw = 20

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
    for i,turbine in enumerate(fi.floris.farm.turbines):
        turbine_powers[i].append(turbine.power/1e6)
    
    # Steady-state version
    fi.calculate_wake(yaw_angles=yaw_angles, axial_induction=ai_values)
    true_powers.append(fi.get_farm_power()/1e6)
    for i,turbine in enumerate(fi.floris.farm.turbines):
        true_turb_powers[i].append(turbine.power/1e6)
    

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)

# Power of individual turbines

plt.figure()
colors = ['C0', 'C1', 'C2']
for i, (turbine_power, tt_pow) in \
    enumerate(zip(turbine_powers, true_turb_powers)):
    label = "Turbine " + str(i)
    plt.plot(list(range(total_time)), turbine_power, label=label)
    plt.plot(list(range(total_time)), tt_pow, label=label, linestyle='dashed')
plt.ylabel("Power (MW)")
plt.xlabel("Time (s)")
plt.legend()

# Total power

plt.figure()

plt.plot(list(range(total_time)), powers, label="Dynamic", color='C1')
plt.plot(list(range(total_time)), true_powers, label="Steady-State", 
    color='C1', linestyle='dashed')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Power (MW)")

plt.show()
