# Project-specific imports
import a2e2g.modules.control.RL.Cluster as cluster
from a2e2g.modules.control.resilience import ResilientWind

# NREL code imports
from floris.tools.optimization.scipy.yaw import YawOptimization
from floris import tools as wfct

# General package imports
import numpy as np
import pandas as pd
from scipy import interpolate



class Baseline():
    """
    Greedy control, tries to keep turbines pointing upwind.
    """
    
    def __init__(self, fi):
        """
        Constructor.
        """
        self.fi = fi
        self.N = len(self.fi.layout_x)
        
        # For use in rated operation
        self.P_t_rated = max(self.fi.get_power_curve(range(12,50)))
        self.Cp_curve = fi.floris.farm.flow_field.turbine_map.turbines[0].\
            fCpInterp
        self.efficiency = max(self.Cp_curve(np.linspace(0,15,100)))/(16/27)
        self.ai_prev = [0.33]*self.N # Initialize

    def step_controller(self, ws, wd, agc, **kwargs):
        """
        Produce control signals based on controller inputs.
        agc not used; ws, wd only used to generate appropriate axial 
        induction factors to pass to WindSE.
        """
        yaw_misalignment_commands = [0.]*self.N
        power_commands = [self.P_t_rated]*self.N
        estimated_power = np.sum(kwargs["P_turbs"])
        
        # Use FLORIS to provide appropriate axial induction factors
        # TODO: If i pass in the actual yaw positions, I could get more 
        # precise axial induction factors.
        #self.fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
        #self.fi.calculate_wake() # No notion of current positions (COULD ALTER)
        #estimated_power = self.fi.get_farm_power()
        
        # P_turbs = kwargs["P_turbs"]
        # axial_induction_commands = [0]*self.N
        
        # # Above-rated operation does not work very well; simulation results
        # # questionable when many turbines are above rated.
        # for t in range(self.N):
        #     if P_turbs[t] >= self.P_t_rated: 
        #         # Turbine producing more than rated power; curtail to rated
        #         Cp_des = self.Cp_curve(ws) # ASSUMES NO WAKES! 
        #         # ^ May cause oscillations
        #         Cp = self.efficiency*4*self.ai_prev[t]*(1-self.ai_prev[t])**2
        #         Cp_des = Cp * self.P_t_rated/P_turbs[t]
                
        #         roots = np.roots([4,-8,4,-Cp_des/self.efficiency])
        #         axial_induction_commands[t] = np.real(roots[2])
        #     else: # Produce maximum power
        #         axial_induction_commands[t] = 0.33
        #     # Provide some smoothing
        #     axial_induction_commands[t] = 0.9*self.ai_prev[t] + \
        #         0.1*axial_induction_commands[t]
        # self.ai_prev = axial_induction_commands
        
        return yaw_misalignment_commands, power_commands, \
            estimated_power

class WakeSteering():
    """
    Wake steering, static, tries to maximize farm power.
    """
    
    def __init__(self, fi, use_lookup_table=False, lookup_file=None, 
        yaw_limits=[0.0, 25.0]):
        """
        Constructor.
        """
        self.lookup = use_lookup_table

        if self.lookup:
            # Load up the lookup table; build the interpolation function
            # WILL need to deal with angle wrapping here!! Could just add 
            # a final 360 one onto the end!
            df_opt = pd.read_pickle(lookup_file)
            self.lu_wds = df_opt.wd.to_list()
            self.lu_wds = self.lu_wds + [360.] # For edge case
            lu_array = np.stack(df_opt.yaw) #axis 0 wd, axis 1 turb
            lu_array = np.vstack((lu_array, lu_array[0,:])) # For edge case
            self.N = lu_array.shape[1]
            
            self.lookup_f = interpolate.interp1d(
                self.lu_wds, lu_array, axis=0, kind='linear'
            )
            self.fi = fi # Still need floris to get aI values
        else:
            self.fi = fi 
            self.N = len(self.fi.layout_x)
            self.min_yaw = yaw_limits[0]
            self.max_yaw = yaw_limits[1]

        # For use in rated operation
        self.P_t_rated = max(self.fi.get_power_curve(range(12,50)))
        self.Cp_curve = fi.floris.farm.flow_field.turbine_map.turbines[0].\
            fCpInterp
        self.efficiency = max(self.Cp_curve(np.linspace(0,15,100)))/(16/27)
        self.ai_prev = [0.33]*self.N # Initialize

    def step_controller(self, ws, wd, agc, **kwargs):
        """
        Produce control signals based on controller inputs.
        agc not used; ws, wd used to determine power-maximizing yaw 
        offsets.
        """
        if self.lookup:
            yaw_misalignment_commands = list(self.lookup_f(wd))
        else:
            self.fi.reinitialize_flow_field(
                wind_direction=wd, wind_speed=ws)

            yaw_opt = YawOptimization(
                self.fi, 
                minimum_yaw_angle=self.min_yaw, 
                maximum_yaw_angle=self.max_yaw
            )
            yaw_misalignment_commands = yaw_opt.optimize()
        
        # Hack to check activating and deactivating wake steering when not 
        # power tracking
        if agc == -1:
            yaw_misalignment_commands = [0.]*self.N 

        # Extract axial_inductions
        self.fi.calculate_wake(yaw_angles=yaw_misalignment_commands)
        estimated_power = self.fi.get_farm_power()
        
        # Above-rated operation does not work very well; simulation results
        # questionable when many turbines are above rated.
        P_turbs = kwargs["P_turbs"]
        axial_induction_commands = [0]*self.N
        for t in range(self.N):
            if P_turbs[t] >= self.P_t_rated: 
                # Turbine producing more than rated power; curtail to rated
                Cp_des = self.Cp_curve(ws) # ASSUMES NO WAKES! 
                # ^ May cause oscillations
                Cp = self.efficiency*4*self.ai_prev[t]*(1-self.ai_prev[t])**2
                Cp_des = Cp * self.P_t_rated/P_turbs[t]
                
                roots = np.roots([4,-8,4,-Cp_des/self.efficiency])
                axial_induction_commands[t] = np.real(roots[2])
            else: # Produce maximum power
                axial_induction_commands[t] = 0.33
            # Provide some smoothing
            axial_induction_commands[t] = 0.9*self.ai_prev[t] + \
                0.1*axial_induction_commands[t]
        self.ai_prev = axial_induction_commands
            
        return yaw_misalignment_commands, axial_induction_commands, \
            estimated_power

class ActivePowerControl():
    """
    a2e2g controller that uses FLORIS to optimize the axial induction 
    and yaw position of turbines to follow a regulation signal from 
    the market.

    Open-loop approach.

    NO LONGER IN USE; NOT SUPPORTED
    """

    def __init__(self, fi, wake_steering_lookup_file=None):
        """
        Constructor.
        """
        # Replace standard floris interface with special version
        fi_s = wfct_special.floris_interface.FlorisInterface(fi.input_file)
        fi_s.reinitialize_flow_field(layout_array=(fi.layout_x,fi.layout_y))
        self.fi = fi_s # Used to extract axial induction factors.
        self.N = len(self.fi.layout_x)
        self.ai_optimizer = ResilientWind(self.fi)

        if wake_steering_lookup_file != None:
            self.using_wake_steering = True
            # Load up the lookup table; build the interpolation function
            df_opt = pd.read_pickle(wake_steering_lookup_file)
            self.lu_wds = df_opt.wd.to_list()
            self.lu_wds = self.lu_wds + [360.] # For edge case
            lu_array = np.stack(df_opt.yaw) #axis 0 wd, axis 1 turb
            lu_array = np.vstack((lu_array, lu_array[0,:])) # For edge case
            self.N = lu_array.shape[1]
            
            self.lookup_f = interpolate.interp1d(
                self.lu_wds, lu_array, axis=0, kind='linear'
            )
        else:
            self.using_wake_steering = False
    
    def step_controller(self, ws, wd, agc, **kwargs):
        """
        Produce control signals to follow agc reference signal based on
        current ws and wd.
        kwargs MUST contain "previous_ai_commands" field for this 
        controller. None is a valid value for this though.
        """
        if self.using_wake_steering: # axial induction on top of wake steering
            yaw_misalignment_commands = list(self.lookup_f(wd))
        else: # Using axial induction only
            yaw_misalignment_commands = [0.]*self.N 
        agc_W = agc*1e6
        axial_induction_commands, estimated_power = \
            self.ai_optimizer.optimize(ws, wd, agc_W, 
                x0=kwargs["previous_ai_commands"],
                yaw_angles=yaw_misalignment_commands)

        return yaw_misalignment_commands, axial_induction_commands, \
            estimated_power

class PIPowerControl():
    """
    Feedback-based power control, based on van-Wingerden et al.
    (2017, https://doi.org/10.1016/j.ifacol.2017.08.378)

    Requires FLORIS for turbine power curves.
    """

    def __init__(self, fi, dt=4, beta=0.7, omega_n=0.01, K_p_reduction=1.0,
        K_i_reduction=0.2, use_battery=False, recharge_threshold=0.95, 
        wake_steering_parameters=None):
        """
        Constructor.

        if wake steering boosting desired, wake_steering_parameters 
        should be a dictionary containing:
        - use_lookup_table=True,
        - lookup_file=None, 
        - yaw_limits=[0.0, 25.0] (only used if use_lookup_table=False)
        - activation_delay=300
        """
        
        # Compute base gains
        self.N = len(fi.layout_x)
        self.K_p = K_p_reduction * 1/self.N
        self.K_i = K_i_reduction*(4*beta*omega_n)
        
        # Controller parameters
        self.dt = dt
        self.Cp_curve = fi.floris.farm.flow_field.turbine_map.turbines[0].\
            fCpInterp
        self.P_t_rated = max(fi.get_power_curve(range(12,50)))
        self.efficiency = max(self.Cp_curve(np.linspace(0,15,100)))/(16/27)
        self.fi = fi
        self.use_battery = use_battery
        if wake_steering_parameters is not None:
            self.use_wake_steering = True
        else:
            self.use_wake_steering = False

        # Initial state for topping off logic
        if self.use_battery:
            self.prevent_charging = False
            self.recharge_threshold = recharge_threshold

        # Set up wake steering
        if self.use_wake_steering:
            # WARNING: not implemented thoroughly---may be issues with lookup 
            # method
            if wake_steering_parameters["use_lookup_table"]:
                self.lookup = True
                df_opt = pd.read_pickle(
                    wake_steering_parameters["lookup_file"])
                self.lu_wds = df_opt.wd.to_list()
                self.lu_wds = self.lu_wds + [360.] # For edge case
                lu_array = np.stack(df_opt.yaw) #axis 0 wd, axis 1 turb
                lu_array = np.vstack((lu_array, lu_array[0,:])) # For edge case
                self.N = lu_array.shape[1]
                
                self.lookup_f = interpolate.interp1d(
                self.lu_wds, lu_array, axis=0, kind="linear"
                )
            else:
                # NOT TESTED
                self.lookup = False
                self.min_yaw = wake_steering_parameters["yaw_limits"][0]
                self.max_yaw = wake_steering_parameters["yaw_limits"][1]
            self.wake_steering_activation_delay = \
                wake_steering_parameters["activation_delay"]
            self.low_power_timer = 0
            self.high_power_timer = 0
            self.wake_steering_active = False # Initially not steering

        # Initialize controller
        self.e_prev = 0
        self.u_prev = 0
        self.u_i_prev = 0
        self.ai_prev = [0.33]*self.N
        self.N_S = 0 

        self.t = 0 # count steps
        
        
    def step_controller(self, ws, wd, agc, **kwargs):
        """
        Produce axial induction factors based on the error between the 
        agc command and the current power.

        kwargs MUST contain "P_turbs" field for this controller. 
        wd input not used by this controller, but is left as an input 
        for consistency with other controllers. It can be set to None.
        """

        # Controller input error
        self.P_turbs = kwargs["P_turbs"]
        e = agc*1e6 - sum(self.P_turbs)

        # If space in battery, adjust the reference
        if self.use_battery:
            b_SOC = kwargs["battery_SOC"]
            b_capacity = kwargs["battery_capacity"]
            b_rate_max = kwargs["battery_rate_limit"]
            b_available = b_capacity - b_SOC
            
            # Logic to prevent excessive topping off
            if b_available/b_capacity > 1-self.recharge_threshold:
                self.prevent_charging = False
            elif b_available/b_capacity <= 0.0001: # Essentially full
                self.prevent_charging = True
            else:
                pass # Use previous value of self.prevent_charging
            
            if self.prevent_charging:
                P_extra = 0
            else:
                P_extra = b_available/self.dt
                P_extra = min(P_extra, b_rate_max)
            
            e = e + P_extra

        # Determine current controller gains
        self.N_S = 0 # TODO: determine whether to use gain scheduling
        if self.N_S < self.N:
            gain_adjustment = self.N/(self.N-self.N_S)
        else:
            gain_adjustment = self.N
        K_p_gs = gain_adjustment*self.K_p
        K_i_gs = gain_adjustment*self.K_i

        # Discretize and apply difference equation (trapezoid rule)
        u_p = K_p_gs*e
        u_i = self.dt/2*K_i_gs * (e + self.e_prev) + self.u_i_prev
        
        # Apply integral anti-windup
        eps = 0.0001 # Threshold for anti-windup
        if (np.array(self.ai_prev) > 1/3-eps).all() or \
           (np.array(self.ai_prev) < 0+eps).all():
           u_i = 0
        u = u_p + u_i

        delta_P_ref = u

        power_commands = self.P_turbs + delta_P_ref
            
        if self.use_wake_steering:
            yaw_misalignment_commands = self.get_wake_steering_commands(
                agc*1e6-sum(self.P_turbs), wd, ws)
        else:
            yaw_misalignment_commands = [0.]*self.N
        
        estimated_power = sum(self.P_turbs) + delta_P_ref*(self.N - self.N_S)

        # Store error, control
        self.e_prev = e
        self.u_prev = u
        self.u_i_prev = u_i
        self.t = self.t + 1

        return yaw_misalignment_commands, power_commands, \
            estimated_power

    # def update_turbine_axial_inductions(self, delta_P_ref, ws):
        
    #     # Update the axial induction factors for each turbine
    #     N_S = 0
    #     self.P_turbs_est = [0]*self.N
    #     axial_induction_commands = [0]*self.N
    #     for t in range(self.N):
    #         if self.ai_prev[t] > 0:
    #             Cp = self.efficiency*4*self.ai_prev[t]*(1-self.ai_prev[t])**2
    #             P_avail = self.efficiency*(16/27) * 1/Cp * self.P_turbs[t]
    #         else: # resort to wind speed to estimate the available power
    #             P_avail = self.fi.get_power_curve([ws])[0]
    #         P_avail = max(P_avail, 0)

    #         if delta_P_ref + self.P_turbs[t] >= self.P_t_rated: 
    #             # Turbine being asked to produce more than rated power
    #             Cp_des = self.Cp_curve(ws) # ASSUMES NO WAKES! 
    #             # ^ May cause oscillations
    #             roots = np.roots([4,-8,4,-Cp_des/self.efficiency])
    #             axial_induction_commands[t] = np.real(roots[2])
    #             N_S = N_S + 1
    #             self.P_turbs_est[t] = self.P_t_rated
    #         elif delta_P_ref + self.P_turbs[t] >= P_avail:
    #             # More power being asked for than available
    #             axial_induction_commands[t] = 1/3 # operate at maximum Cp
    #             N_S = N_S + 1
    #             self.P_turbs_est[t] = P_avail
    #         elif delta_P_ref + self.P_turbs[t] <= 0:
    #             # Asking for negative power production
    #             axial_induction_commands[t] = 0 # Shut off turbine
    #             self.P_turbs_est[t] = 0
    #             N_S = N_S + 1 # Turbine can no longer follow reference
    #         else:
    #             # Case where there is sufficient overhead to follow command
    #             Cp_des = self.efficiency * (16/27) * \
    #                 ((delta_P_ref + self.P_turbs[t])/P_avail)
    #             roots = np.roots([4,-8,4,-Cp_des/self.efficiency])
    #             axial_induction_commands[t] = np.real(roots[2])
    #             self.P_turbs_est[t] = delta_P_ref + self.P_turbs[t]
    #             # No update to N_S, since turbine able to follow reference.
    #     self.N_S = N_S
    #     self.ai_prev = axial_induction_commands

    #     return axial_induction_commands

    def get_wake_steering_commands(self, farm_power_error, wd, ws):
        
        # Step counters
        if farm_power_error > 0: # underproducing
            self.low_power_timer += 1*self.dt
        else:
            self.low_power_timer = 0
            if self.wake_steering_active:
                self.high_power_timer += 1*self.dt
            else:
                self.high_power_timer = 0

        if self.low_power_timer >= self.wake_steering_activation_delay:
            self.wake_steering_active = True
        elif self.high_power_timer >= self.wake_steering_activation_delay:
            self.wake_steering_active = False
        else:
            pass # No change to wake steering activation state

        if self.wake_steering_active:
            if self.lookup:
                yaw_misalignment_commands = list(self.lookup_f(wd))
            else:
                self.fi.reinitialize_flow_field(
                    wind_direction=wd, wind_speed=ws)
                yaw_opt = YawOptimization(
                    self.fi, 
                    minimum_yaw_angle=self.min_yaw, 
                    maximum_yaw_angle=self.max_yaw
                )
                yaw_misalignment_commands = yaw_opt.optimize()
        else:
            yaw_misalignment_commands = [0.]*self.N
        
        return yaw_misalignment_commands

class IndividualAPC():
    """
    Open-loop power control, based on Fleming (?) et al.

    Requires FLORIS for turbine power curves.
    """

    def __init__(self, fi, dt=4):
        """
        Constructor.
        """
        
        # Parameters
        self.N = len(fi.layout_x)
        self.dt = dt
        self.Cp_curve = fi.floris.farm.flow_field.turbine_map.turbines[0].\
            fCpInterp
        self.P_t_rated = max(fi.get_power_curve(range(12,50)))
        self.efficiency = max(self.Cp_curve(np.linspace(0,15,100)))/(16/27)
        self.fi = fi

        # Initialize controller
        self.ai_prev = [0.33]*self.N

        self.t = 0 # count steps

    def step_controller(self, ws, wd, agc, **kwargs):
        """
        Produce axial induction factors based on the error between the 
        agc command and the current power, divided among the turbines.

        kwargs MUST contain "P_turbs" field for this controller. 
        wd input not used by this controller, but is left as an input 
        for consistency with other controllers. It can be set to None.
        """

        # Controller input error for individual turbines
        self.P_turbs = kwargs["P_turbs"]
        power_commands = [agc*1e6/self.N]*self.N
            
        yaw_misalignment_commands = [0.]*self.N
        estimated_power = sum(self.P_turbs_est) # TODO: Update this.

        # Store error, control
        self.t = self.t + 1

        return yaw_misalignment_commands, power_commands, \
            estimated_power

    def update_turbine_axial_inductions(self, P_ref_turb, ws):
        
        # Update the axial induction factors for each turbine
        axial_induction_commands = [0]*self.N
        self.P_turbs_est = [0]*self.N
        for t in range(self.N):
            if self.ai_prev[t] > 0:
                Cp = self.efficiency*4*self.ai_prev[t]*(1-self.ai_prev[t])**2
                P_avail = self.efficiency*(16/27) * 1/Cp * self.P_turbs[t]
            else: # resort to wind speed to estimate the available power
                P_avail = self.fi.get_power_curve([ws])[0]
            P_avail = max(P_avail, 0)

            if P_ref_turb >= self.P_t_rated: 
                # Turbine being asked to produce more than rated power
                Cp_des = self.Cp_curve(ws) # ASSUMES NO WAKES! 
                # ^ May cause oscillations
                roots = np.roots([4,-8,4,-Cp_des/self.efficiency])
                axial_induction_commands[t] = np.real(roots[2])
                self.P_turbs_est[t] = self.P_t_rated
            elif P_ref_turb >= P_avail:
                # More power being asked for than currently available
                axial_induction_commands[t] = 1/3 # operate at maximum Cp
                self.P_turbs_est[t] = P_avail
            elif P_ref_turb <= 0:
                # Asking for negative power production
                axial_induction_commands[t] = 0 # Shut off turbine
                self.P_turbs_est[t] = 0
            else:
                # Case where there is sufficient overhead to follow command
                Cp_des = self.efficiency * (16/27) * (P_ref_turb/P_avail)
                roots = np.roots([4,-8,4,-Cp_des/self.efficiency])
                axial_induction_commands[t] = np.real(roots[2])
                self.P_turbs_est[t] = P_ref_turb
        self.ai_prev = axial_induction_commands

        return axial_induction_commands

def generate_wake_steering_lookup_table(fi, N_wds=360, yaw_limits=[0.0, 25.0]):
    """
    Sweep through wind speeds to generate optimal controls at 8 m/s.
    """

    wind_directions = np.linspace(0, 360-360/N_wds, N_wds)

    ws = 8.0
    optimal_yaws = []
    for wd in wind_directions:

        print('Wind direction: {0:.2f} degrees.'.format(wd))
        fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
 
        yaw_opt = YawOptimization(
            fi, 
            minimum_yaw_angle=yaw_limits[0], 
            maximum_yaw_angle=yaw_limits[1]
        )
        optimal_yaws.append(yaw_opt.optimize())

    return optimal_yaws, wind_directions

if __name__ == "__main__":
    fi = wfct.floris_interface.FlorisInterface(
        "./a2e2g/modules/control/datasets/example_input.json"                   
    )
    layout = pd.read_csv(
        "./a2e2g/modules/control/datasets/layout.csv"
    )
    h = cluster.Cluster(layout)
    layout_x, layout_y = h.getFarmCluster() 
    fi.reinitialize_flow_field(layout_array=(layout_x,layout_y))
    
    opt_yaws, wind_dir = generate_wake_steering_lookup_table(fi, N_wds=180)

    df_opt = pd.DataFrame({'wd':wind_dir, 'yaw':opt_yaws})
    df_opt.to_pickle('./a2e2g/modules/control/wake_steering_offsets.pkl')