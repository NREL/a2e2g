
import numpy as np

from numpy import linspace, zeros, exp
import matplotlib.pyplot as plt
import random
from scipy.stats import norm,multivariate_normal




def reset(fi,no_turb,wnd_init, ref):

    
    '''
    Initialize yaws to 0
    '''
    fi.reinitialize_flow_field(wind_speed=[wnd_init],sim_time = 0)
    
    Wnd = np.zeros(no_turb+1,)
    fi.calculate_wake(yaw_angles=[0]*no_turb)
    for wnd_i in range(no_turb):
        Wnd[wnd_i] = fi.floris.farm.turbines[wnd_i].average_velocity;
    Wnd[-1] = ref
    return Wnd, fi


        

def step(a,x,fi,sim_time,pwr_4s, wnd_4s, no_turb):
    
    '''
    This function contains the dyanmics of the wake
    
    a :: action to be chosen
    x :: state (wind speed)
    fi :: floris interface object
    sim_time :: time of simulation
    
    
    '''

    a_fil = a
    
    
    # Set power to zero if wind too low
    if wnd_4s < 2 or pwr_4s <1e-4:
        power = 0;
        pwr_4s = 0 # Also set command to 0 to avoid a bug
    else:
        fi.reinitialize_flow_field(wind_speed=[wnd_4s],sim_time = sim_time)
        #fi.change_turbine(list(range(no_turb)), {"a": a_fil[no_turb:]})
        fi.calculate_wake(yaw_angles=a_fil[:no_turb])
        pow_array = fi.get_turbine_power()
        power = np.sum(pow_array)
    
    # State update
    
    # number of states + agc signal
    Wnd = np.zeros(no_turb+1,)
    for wnd_i in range(no_turb):
        Wnd[wnd_i] = fi.floris.farm.turbines[wnd_i].average_velocity;
    Wnd[-1] = pwr_4s
    x_next = Wnd;
    

    
    # Rewards for the system
    rewards = 0;
    low_pow = 0.85*1e6*pwr_4s
    up_pow = 1.15*1e6*pwr_4s
    
    # If power generated is greater than the power output of farm,
    # then neglect few turbines so tracking is easier
    if power > up_pow:
        pow_asc = np.sort(pow_array);
        pow_asc_idx = np.argsort(pow_asc)
        cum_pow = np.cumsum(pow_asc)
        idx = np.abs(pwr_4s*1e6 - cum_pow).argmin()
        power = cum_pow[idx]
        #rewards = .1;
        
    if power <1e-6:
        rewards = 0;
        
    else:
            # rewards reduce as error increases
            rewards = np.exp(-1*np.abs(power*1e-6 - pwr_4s));

        
    
    done = bool(0)
    return x_next, rewards, done, fi, power



