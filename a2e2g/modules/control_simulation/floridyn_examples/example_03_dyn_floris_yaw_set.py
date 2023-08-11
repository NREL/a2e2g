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

from floridyn_special2 import tools as wfct_d # Incoming con
from floris import tools as wfct_s

test_ai_changes = False # Else, test yaw angle change
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
fi_d = wfct_d.floris_interface.FlorisInterface(floris_dir)
fi_s = wfct_s.floris_interface.FlorisInterface(floris_dir)

# Calculate wake
fi_d.calculate_wake()

# Get horizontal plane at default height (hub-height) before simulation
# (not used)
hor_plane_d = fi_d.get_hor_plane()

powers_d = []
powers_s1 = []
powers_s2 = []
total_time = 700
fi_d.floris.farm.flow_field.mean_wind_speed = 8
turbine_powers_d = [ [] for turbine in fi_d.floris.farm.turbines]
turbine_powers_s1 = [ [] for turbine in fi_d.floris.farm.turbines]
turbine_powers_s2 = [ [] for turbine in fi_s.floris.farm.turbines]
# turb_0_yaw = 20

# fi.calculate_wake()

yaw_angles = [0 for turbine in fi_d.floris.farm.turbines]
ai_values = [0.33 for turbine in fi_d.floris.farm.turbines]
fi_d.calculate_wake(axial_induction=ai_values, yaw_angles=yaw_angles)
fi_s.calculate_wake(yaw_angles=yaw_angles) # Can't pass axial inductions into
# standard FLORIS!

for sim_time in range(total_time):
    if sim_time % 100 == 0:
        print("Time step:", sim_time)
    
    if test_ai_changes:
        if sim_time in ai_changes:
            ai_values = ai_changes[sim_time]
    else:
        if sim_time in angle_changes:
            yaw_angles = angle_changes[sim_time]
        #fi.reinitialize_flow_field(wind_speed=10, sim_time=sim_time)
    
    # Try to look at a flow field part way through the simulation
    if sim_time == 600:
        hor_plane_d = fi_d.get_hor_plane()
    
    #if test_ai_changes:
    fi_d.calculate_wake(sim_time=sim_time, yaw_angles=yaw_angles, 
        axial_induction=ai_values)
    powers_d.append(fi_d.get_farm_power()/1e6)
    for i,turbine in enumerate(fi_d.floris.farm.turbines):
        turbine_powers_d[i].append(turbine.power/1e6)
    
    # Steady-state version 
    # *** DOES NOT SEEM TO WORK AS EXPECTED ***
    fi_d.calculate_wake(yaw_angles=yaw_angles, axial_induction=ai_values)
    powers_s1.append(fi_d.get_farm_power()/1e6)
    for i,turbine in enumerate(fi_d.floris.farm.turbines):
        turbine_powers_s1[i].append(turbine.power/1e6)
    if sim_time == 600:
        hor_plane_s1 = fi_d.get_hor_plane()

    # New steady-state version
    fi_s.calculate_wake(yaw_angles=yaw_angles) # Can't pass axial inductions 
    # into standard FLORIS!
    powers_s2.append(fi_s.get_farm_power()/1e6)
    for i,turbine in enumerate(fi_s.floris.farm.turbines):
        turbine_powers_s2[i].append(turbine.power/1e6)
    if sim_time == 600:
        hor_plane_s2 = fi_s.get_hor_plane()

    

# Plot and show
fig, ax = plt.subplots()
wfct_d.visualization.visualize_cut_plane(hor_plane_d, ax=ax)
ax.set_title('Dynamic')

fig, ax = plt.subplots()
wfct_d.visualization.visualize_cut_plane(hor_plane_s1, ax=ax)
ax.set_title('Steady-state from dynamic')

fig, ax = plt.subplots()
wfct_d.visualization.visualize_cut_plane(hor_plane_s2, ax=ax)
ax.set_title('Steady-state from FLORIS')


# Power of individual turbines

plt.figure()
colors = ['C0', 'C1', 'C2']
for i, (Pturb_d, Pturb_s1, Pturb_s2) in \
    enumerate(zip(turbine_powers_d, turbine_powers_s1, turbine_powers_s2)):
    label = "Turbine " + str(i)
    plt.plot(list(range(total_time)), Pturb_d, label=label)
    plt.plot(list(range(total_time)), Pturb_s1, label=label, linestyle='dashed')
    plt.plot(list(range(total_time)), Pturb_s2, label=label, linestyle='dotted')
plt.ylabel("Power (MW)")
plt.xlabel("Time (s)")
plt.legend()

# Total power

plt.figure()

plt.plot(list(range(total_time)), powers_d, label="Dynamic", color='C1')
plt.plot(list(range(total_time)), powers_s1, label="SS from dynamic", 
    color='C1', linestyle='dashed')
plt.plot(list(range(total_time)), powers_s2, label="SS from FLORIS", 
    color='C1', linestyle='dotted')
initial_diff = powers_d[0] - powers_s2[0]
plt.plot(list(range(total_time)), powers_s2+initial_diff, 
    label="SS from FLORIS, adjusted", 
    color='black', linestyle='dotted')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Power (MW)")

plt.show()
