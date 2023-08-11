import numpy as np
import pandas as pd
import json, yaml
from scipy import signal

from a2e2g.modules.control_simulation.floridyn_special2 import tools as wfct_floridyn

# Import a floriDyn version too

class Simulator():
    """
    Parent class for simulating with WindSE (not supported!), static 
    floris, or dynamic floris.
    """

    def __init__(self, local_dir, 
                absolute_yaw_ics=0, yaw_ics_range=0, 
                axial_induction_ics=0.33, 
                dt=4.0, yaw_rate=0.5, yaw_deadband=8.0, 
                axial_induction_rate_limit=1.0, 
                axial_induction_time_constant=0.1,
                axial_induction_control_natural_frequency=None,
                axial_induction_control_damping_ratio=None,
                N=1):

        self.dir = local_dir

        # Initialize yaw angle and axial induction (only needed once 
        # limit_control_inputs is built out)
        if np.isscalar(absolute_yaw_ics):
            absolute_yaw_ics = list(absolute_yaw_ics + \
                np.random.uniform(low=-yaw_ics_range/2, high=yaw_ics_range/2, 
                                  size=N)
            )
            
        if np.isscalar(axial_induction_ics):
            axial_induction_ics = [axial_induction_ics]*N

        self.yaw_rate = yaw_rate
        self.deadband = yaw_deadband
        self.ai_rate_limit = axial_induction_rate_limit
        self.dt = dt
        self.yaw_state = [0]*N # All turbines start out not yawing
        
        # Establish axial induction dynamic model
        self._establish_ai_model(
            axial_induction_time_constant,
            axial_induction_control_natural_frequency,
            axial_induction_control_damping_ratio
        )

        self._set_initial_conditions(absolute_yaw_ics, axial_induction_ics)

    def _establish_ai_model(self, axial_induction_time_constant,
            axial_induction_control_natural_frequency,
            axial_induction_control_damping_ratio):

        if axial_induction_control_natural_frequency is not None and \
            axial_induction_control_natural_frequency is not None:
            using_control_model = True
        elif axial_induction_control_natural_frequency is None and \
            axial_induction_control_natural_frequency is None:
            using_control_model = False
        else:
            raise ValueError("Either define both natural frequence and "+\
                "damping, or set both to None.")

        if using_control_model:
            # For ease of use
            omega_n = axial_induction_control_natural_frequency
            zeta = axial_induction_control_damping_ratio
            
            sys_turb_ct = signal.TransferFunction([omega_n**2],
                [1, 2*zeta*omega_n, omega_n**2]
            )
        
        ai_tau = axial_induction_time_constant
        sys_flow_ct = signal.TransferFunction([1/ai_tau],[1, 1/ai_tau])
        
        # Discretize, convert to statespace models
        sys_turb_dt = sys_turb_ct.to_discrete(self.dt, method="bilinear")
        A_turb_ss, B_turb_ss, C_turb_ss, D_turb_ss = \
            signal.tf2ss(sys_turb_dt.num, sys_turb_dt.den)
        sys_flow_dt = sys_flow_ct.to_discrete(self.dt, method="bilinear")
        A_flow_ss, B_flow_ss, C_flow_ss, D_flow_ss = \
            signal.tf2ss(sys_flow_dt.num, sys_flow_dt.den)

        # unpack result and save for later use
        self.turb_reponse_mdl = {
            "A":A_turb_ss, "B":B_turb_ss, "C":C_turb_ss, "D":D_turb_ss
        }
        self.flow_reponse_mdl = {
            "A":A_flow_ss, "B":B_flow_ss, "C":C_flow_ss, "D":D_flow_ss
        }

        return None

    def _set_initial_conditions(self, absolute_yaw_ics, axial_induction_ics):
        """
        Establish initial conditions for yaw and axial induction for 
        all turbines.
        """

        self.absolute_yaw_positions = absolute_yaw_ics
        self.axial_induction_factors = axial_induction_ics
        Cp_init = 0.7*4*np.array(axial_induction_ics)*\
            (1-np.array(axial_induction_ics))**2 
        # Imperfect efficiency

        n_x_cp_turb = self.turb_reponse_mdl["A"].shape[0]

        self.x_cp_turb = [np.ones((n_x_cp_turb, 1)) * \
            (1-self.turb_reponse_mdl["D"][0,0])*\
                cp_ic/self.turb_reponse_mdl["C"].sum()
            for cp_ic in Cp_init]

        n_x_ai_flow = self.flow_reponse_mdl["A"].shape[0]

        self.x_ai_flow = [np.ones((n_x_ai_flow, 1)) * \
            (1-self.flow_reponse_mdl["D"][0,0])*\
                ai_ic/self.flow_reponse_mdl["C"].sum()
            for ai_ic in axial_induction_ics]
        

        return None

    def limit_control_inputs(self, yaw_commands, axial_induction_commands, ws, 
                             wd):
        """
        Constrain the control inputs to realistic values.

        yaw_commands and axial_induction_commands should be lists of 
        length (N) (where N is the number of turbines). 
        
        yaw_commands is taken to be relative to the wind direction.
        ws not currently used (for later development).
        """

        # Compute vane signal (w/ possible offset)
        absolute_yaw_commands = wd - np.array(yaw_commands) # bearing angle
        yaw_diff_commands = absolute_yaw_commands - \
                            np.array(self.absolute_yaw_positions)
        yaw_diff_commands = [wrap_to_180(ydc) for ydc in yaw_diff_commands]
        
        for t in range(len(self.absolute_yaw_positions)):
            if self.yaw_state[t] == 0: # Not currently yawing
                if yaw_diff_commands[t] >= self.deadband:
                    self.yaw_state[t] = 1 # Start yawing to reduce pos error
                elif yaw_diff_commands[t] <= -self.deadband:
                    self.yaw_state[t] = -1 # Start yawing to reduce neg error
                else: # Within deadband
                    pass # Do not initiate yawing
            elif self.yaw_state[t] == 1: # Yawing to reduce positive error
                if yaw_diff_commands[t] <= 0: # Crosses zero
                    self.yaw_state[t] = 0 # Stop yawing
            elif self.yaw_state[t] == -1: # Yawing to reduce negative error
                if yaw_diff_commands[t] >= 0: # Crosses zero
                    self.yaw_state[t] = 0 # Stop yawing

        # Take yaw action, if necessary
        # self.absolute_yaw_positions = self.absolute_yaw_positions + \
        #     self.yaw_state * self.yaw_rate*self.dt
        self.absolute_yaw_positions = [
            wrap_to_360(ayp + ys*self.yaw_rate*self.dt) for \
            ayp, ys in zip(self.absolute_yaw_positions, self.yaw_state)
        ]

        # Constrain axial inductions
        ai_diff_commands = np.array(axial_induction_commands) - \
                           np.array(self.axial_induction_factors)
        ai_diffs = np.clip(ai_diff_commands, -self.ai_rate_limit*self.dt, 
                           self.ai_rate_limit*self.dt)
        steady_ai_factor = np.clip(self.axial_induction_factors+ai_diffs, 
            0.0, 0.33)

        # Dynamic corrections
        # (see Knudsen and Bak, 2013 and Hansen et al., 2005)
        dynamic_ai_factors = [0]*steady_ai_factor.shape[0]
        for t, sai in enumerate(steady_ai_factor):
            dynamic_ai_factors[t] = (self.ai_mdl["C"] @ self.x_ai[t] + \
                                     self.ai_mdl["D"] * sai)[0,0]
            self.x_ai[t] = (self.ai_mdl["A"] @ self.x_ai[t] + \
                            self.ai_mdl["B"] * sai)            

        self.axial_induction_factors = dynamic_ai_factors
        
        return None
        
    
    def turbine_power_response(self, turbine_power_commands, ws):
        """
        Not called by baseline control.
        """
        
        # Update the axial induction factors for each turbine
        num_saturated_turbs = 0
        self.P_turbs_est = [0]*self.N

        turbine_axial_inductions = [0]*self.N
        for t in range(self.N):
            if self.ai_prev[t] > 0:
                Cp = self.efficiency*4*self.ai_prev[t]*(1-self.ai_prev[t])**2
                P_avail = self.efficiency*(16/27) * 1/Cp * self.P_turbs[t]
            else: # resort to wind speed to estimate the available power
                #P_avail = self.fi.get_power_curve([ws])[0]
                P_avail = 0.5 * self.fi.floris.farm.turbines[0].air_density * \
                   ws**3 * np.pi * \
                   self.fi.floris.farm.turbines[0].rotor_radius**2 * \
                   self.Cp_curve(ws)
            P_avail = max(P_avail, 0)

            if turbine_power_commands[t] >= self.P_t_rated: 
                # Turbine being asked to produce more than rated power
                Cp_des = self.Cp_curve(ws) # ASSUMES NO WAKES! 
                # ^ May cause oscillations
                num_saturated_turbs = num_saturated_turbs + 1
                self.P_turbs_est[t] = self.P_t_rated
            elif turbine_power_commands[t] >= P_avail:
                # More power being asked for than available
                Cp_des = self.efficiency * 4*(1/3)*(1-1/3)**2 # maximuma
                num_saturated_turbs = num_saturated_turbs + 1
                self.P_turbs_est[t] = P_avail
            elif turbine_power_commands[t] <= 0:
                # Asking for negative power production
                Cp_des = 0 # Shut off turbine
                self.P_turbs_est[t] = 0
                num_saturated_turbs = num_saturated_turbs + 1 
                # Turbine can no longer follow reference
            else:
                # Case where there is sufficient overhead to follow command
                Cp_des = self.efficiency * (16/27) * \
                    ((turbine_power_commands[t])/P_avail)
                self.P_turbs_est[t] = turbine_power_commands[t]
                # No update to N_S, since turbine able to follow reference.
        
            # Turbine power response
            Cp_actual = self.step_turbine_dynamics(Cp_des, t)
            roots = np.roots([4,-8,4,-Cp_actual/self.efficiency])
            turbine_axial_inductions[t] = np.real(roots[2])


        self.num_saturated_turbs = num_saturated_turbs
        self.ai_prev = turbine_axial_inductions # Store
        self.turbine_axial_inductions = turbine_axial_inductions

        return turbine_axial_inductions

    def step_turbine_dynamics(self, Cp_des_t, t):
        # t is the current turbine
        Cp_actual_t = (
            self.turb_reponse_mdl["C"] @ self.x_cp_turb[t] + \
            self.turb_reponse_mdl["D"] * Cp_des_t
        )[0,0]
        
        self.x_cp_turb[t] = (
            self.turb_reponse_mdl["A"] @ self.x_cp_turb[t] + \
            self.turb_reponse_mdl["B"] * Cp_des_t
        )

        return Cp_actual_t

    def flow_response(self, turbine_axial_inductions):

        flow_axial_inductions = [0.]*self.N

        for t in range(self.N):
        
            flow_axial_inductions[t] = (
                self.flow_reponse_mdl["C"] @ self.x_ai_flow[t] + \
                self.flow_reponse_mdl["D"] * turbine_axial_inductions[t]
            )[0,0]
            
            self.x_ai_flow[t] = (
                self.flow_reponse_mdl["A"] @ self.x_ai_flow[t] + \
                self.flow_reponse_mdl["B"] * turbine_axial_inductions[t]
            )

        self.flow_axial_inductions = flow_axial_inductions
        
        return flow_axial_inductions

    
    def yaw_system_response(self, yaw_commands, wd):
        """
        Constrain the control inputs to realistic values.

        yaw_commands and axial_induction_commands should be lists of 
        length (N) (where N is the number of turbines). 
        
        yaw_commands is taken to be relative to the wind direction.
        ws not currently used (for later development).
        """

        # Compute vane signal (w/ possible offset)
        absolute_yaw_commands = wd - np.array(yaw_commands) # bearing angle
        yaw_diff_commands = absolute_yaw_commands - \
                            np.array(self.absolute_yaw_positions)
        yaw_diff_commands = [wrap_to_180(ydc) for ydc in yaw_diff_commands]
        
        for t in range(len(self.absolute_yaw_positions)):
            if self.yaw_state[t] == 0: # Not currently yawing
                if yaw_diff_commands[t] >= self.deadband:
                    self.yaw_state[t] = 1 # Start yawing to reduce pos error
                elif yaw_diff_commands[t] <= -self.deadband:
                    self.yaw_state[t] = -1 # Start yawing to reduce neg error
                else: # Within deadband
                    pass # Do not initiate yawing
            elif self.yaw_state[t] == 1: # Yawing to reduce positive error
                if yaw_diff_commands[t] <= 0: # Crosses zero
                    self.yaw_state[t] = 0 # Stop yawing
            elif self.yaw_state[t] == -1: # Yawing to reduce negative error
                if yaw_diff_commands[t] >= 0: # Crosses zero
                    self.yaw_state[t] = 0 # Stop yawing

        # Take yaw action, if necessary
        self.absolute_yaw_positions = [
            wrap_to_360(ayp + ys*self.yaw_rate*self.dt) for \
            ayp, ys in zip(self.absolute_yaw_positions, self.yaw_state)
        ]
        
        return self.absolute_yaw_positions

    def simulate_time_series(self, df, verbose=True):
        """
        Simulate the farm over a time series. df is assumed to have 
        the following columns (where N=30 is the number of turbines 
        in the farm): ws (float), wd (float), yaw 
        (list of N floats), axial_induction (list of N floats).
        """

        T = len(df)

        Powers = [0.0]*T
        Yaws = []
        AIs = []
        for i_r, row in enumerate(df.itertuples(index=False, name='step')):
            if verbose:
                if self._type() == 'SimulateFLORIS': # Print sporadically 
                    if i_r % 100 == 0:
                        print('\nSimulating time step {0} of {1}.\n'.\
                            format(i_r+1, T))
                else: # Print every step
                    print('\nSimulating time step {0} of {1}.\n'.\
                        format(i_r+1, T))

            Ptot_t = self.step_simulator(
                row.ws, 
                row.wd, 
                row.yaw, 
                row.axial_induction, 
                verbose=verbose
            )[1]

            Powers[i_r] = Ptot_t

            # Also save the actual axial inductions and yaw angles
            Yaws.append([wrap_to_360(row.wd - abs_yaw) for \
                abs_yaw in self.absolute_yaw_positions])
            AIs.append(self.axial_induction_factors)

        df['P_true'] = Powers 
        df['yaw_actual'] = Yaws
        df['ai_actual'] = AIs

        return df
            
    def _test_limit_control_inputs(self, ws, wd, yaw_commands,
                                   axial_induction_commands):
        """
        For testing that limit_control_inputs is working as desired.
        """

        self.limit_control_inputs(
            yaw_commands, 
            axial_induction_commands, 
            ws, 
            wd
        )
         
        self.generate_input_files(ws, wd, verbose=True)
        
        return None

    def _type(self):
        return self.__class__.__name__

class SimulateWindSE(Simulator):
    """
    Class for running WindSE given a inflow wind speed and direction 
    and commanded yaw angles and axial inductions for all turbines.

    NOT SUPPORTED/TESTED.
    """
    
    def __init__(self, input_file_dir, 
                 absolute_yaw_ics=0, yaw_ics_range=0,
                 axial_induction_ics=0.33, dt=4.0, 
                 yaw_rate=0.5, yaw_deadband=8.0, 
                 axial_induction_rate_limit=1.0, 
                 axial_induction_time_constant=0.1,
                 wd_method='rotate domain', _3d=True, warp_percent=0.9):

        """
        Initializer.

        Inputs:
           local_dir: string - folder where this package is located
           absolute_yaw_ics: float or list - initial conditions for yaw
           axial_induction_ics: float or list - ics for axial induction
           dt: float - time step length [s]
           yaw_rate: float - yawing speed [deg/s]
           ai_rate_limit: float - axial induction rate limit [-/s]
        """
        super().__init__(input_file_dir, absolute_yaw_ics, yaw_ics_range, 
            axial_induction_ics,
            dt, yaw_rate, yaw_deadband, axial_induction_rate_limit, 
            axial_induction_time_constant)

        self.wd_method = wd_method
        self._3d = _3d
        self.warp_percent = warp_percent

    def generate_input_files(self, ws, wd, yaw_positions=None, 
                             axial_induction_factors=None,
                             verbose=True):
        """
        Update the .yaml and .txt file that WindSE needs.
        yaw_positions and axial_induction_factors inputs can be used 
        to override the saved `current' yaws, ais.
        """

        # Override current yaw_angles, axial_induction_factors
        if yaw_positions is not None:
            yaw_misalignments = yaw_positions
        else:
            yaw_misalignments = wd - np.array(self.absolute_yaw_positions)
        
        if axial_induction_factors is not None:
            self.axial_induction_factors = axial_induction_factors

        # Convert yaw_positions and axial_induction_factors to CCW rad.
        yaw_misalignments_rad = [yaw_angle_to_CCW_angle(y) for y in \
                                 yaw_misalignments]
        wd_rad = bearing_to_CCW_angle(wd)
        
        # txt file first
        self.cluster_layout['Yaw'] = yaw_misalignments_rad
        self.cluster_layout['Axial_Induction'] = self.axial_induction_factors

        header_string = '# '+\
                        ' '.join(self.cluster_layout.columns.to_list())+\
                        '\n'
        f = open(self.dir+'/wind_farm.txt', 'w')
        f.write(header_string)
        if self.wd_method == 'inflow angle':
            self.cluster_layout.to_csv(f, sep=' ', index=False, header=False)
        elif self.wd_method == 'rotate domain':
            rotated_layout = self.cluster_layout.copy()
            rotated_layout[['x', 'y']] = self.cluster_layout[['x', 'y']].\
                apply(rotate_domain, args=(wd_rad,), axis=1, 
                      result_type='expand')
            rotated_layout.to_csv(f, sep=' ', index=False, header=False)
        else:
            print('Invalid wind direction method (wd_method) specified.')
        f.close()

        # then yaml file
        if self.wd_method == 'inflow angle':
            if self._3d:
                with open(self.dir+'/wind_farm_template_box.yaml') as f:
                    loaded_yaml = yaml.safe_load(f)
            else:
                with open(self.dir+'/wind_farm_template_rectangle.yaml') as f:
                    loaded_yaml = yaml.safe_load(f)
            loaded_yaml['boundary_conditions']['HH_vel'] = ws
            loaded_yaml['boundary_conditions']['inflow_angle'] = wd_rad
        elif self.wd_method == 'rotate domain':
            if self._3d:
                with open(self.dir+'/wind_farm_template_cylinder.yaml') as f:
                    loaded_yaml = yaml.safe_load(f)
                    loaded_yaml['refine']['warp_percent'] = self.warp_percent
            else:
                with open(self.dir+'/wind_farm_template_circle.yaml') as f:
                    loaded_yaml = yaml.safe_load(f)
            loaded_yaml['boundary_conditions']['HH_vel'] = float(ws)
        loaded_yaml['wind_farm']['path'] = self.dir+'/wind_farm.txt'
        loaded_yaml['general']['output_folder'] = self.dir+'/output/'
        
        with open(self.dir+'/wind_farm.yaml', 'w') as f:
            yaml.dump(loaded_yaml, f)

        # Print statement of changes
        if verbose:
            print(('\nWindSE input files generated.\n'+\
                   '\nWind speed: {0:.1f} m/s\n'+\
                   'Wind direction: {1:.1f} deg\n' \
                    ).format(ws, wd))
            print('\n'.join(('T{0:02d}: abs. yaw pos.: {1:.1f} deg; '+\
                  'a.i. factor: {2:.2f} ').format(*vals) for 
                  vals in zip(range(len(self.absolute_yaw_positions)), 
                              self.absolute_yaw_positions,
                              self.axial_induction_factors)))
            print('\n')


        return None 

    def run_windse(self):
        """
        Perform WindSE steps. Based on WindSE demos.
        """

        ### Create an Instance of the Options ###
        windse.initialize(self.dir+'/wind_farm.yaml') 

        ### Generate Domain ###
        if self.wd_method == 'inflow angle':
            if self._3d:
                dom = windse.BoxDomain()
            else:
                dom = windse.RectangleDomain()
            #dom = windse.BoxDomain()
        elif self.wd_method == 'rotate domain':
            if self._3d:
                dom = windse.CylinderDomain()
            else:
                dom = windse.CircleDomain()
        
        # Are these commands needed?
        if self._3d:
            dom.WarpSplit(250, self.warp_percent)
        dom.Refine(1)

        ### Generate Wind Farm ###
        farm = windse.ImportedWindFarm(dom) 

        ### Function Space ###
        if self._3d:
            fs = windse.LinearFunctionSpace(dom)
        else:
            fs = windse.TaylorHoodFunctionSpace(dom)

        ### Setup Boundary Conditions ###
        if self._3d:
            bc = windse.PowerInflow(dom,fs,farm)
        else:
            bc = windse.UniformInflow(dom,fs,farm)

        ### Generate the problem ###
        if self._3d:
            problem = windse.StabilizedProblem(dom,farm,fs,bc)
        else:
            problem = windse.TaylorHoodProblem(dom,farm,fs,bc)

        ### Solve ###
        solver = windse.SteadySolver(problem)
        solver.Solve()

        ### Output Results ###
        solver.Save()
        
        return None
    
    def get_output_power(self):
    
        f = open(self.dir+'/output/sim/data/power_data.txt', 'r')
        f.readline() # Skip first line
        vals = f.readline().split()
        f.close()

        turbine_powers = [float(v) for v in vals[1:-1]]
        total_power = float(vals[-1])

        return turbine_powers, total_power

    def step_simulator(self, ws, wd, yaw_commands, axial_induction_commands, 
                       verbose=True):
        """
        Perform steps to update control inputs, generate windse input
        files, and run the windse code to produce the output power.
        
        yaw_commands taken to be relative to oncoming winds 
        (i.e., 0 is aligned with the wind direction)
        """

        self.limit_control_inputs(
            yaw_commands, 
            axial_induction_commands, 
            ws, 
            wd
        )
        
        self.generate_input_files(ws, wd, verbose=verbose)

        self.run_windse()

        turbine_powers, total_power = self.get_output_power()

        return turbine_powers, total_power

class SimulateFLORIS(Simulator):
    """
    Class for running FLORIS given a inflow wind speed and direction 
    and commanded yaw angles and axial inductions for all turbines.
    """
    
    def __init__(self, input_file_dir, model_path,
                 absolute_yaw_ics=0, yaw_ics_range=0,
                 axial_induction_ics=0.33, dt=4.0, 
                 yaw_rate=0.5, yaw_deadband=8.0, 
                 axial_induction_rate_limit=1.0, 
                 axial_induction_time_constant=0.1, 
                 axial_induction_control_natural_frequency=None,
                 axial_induction_control_damping_ratio=None,
                 use_FLORIDyn=True):
        """
        Initializer.

        Inputs:
           local_dir: string - folder where this package is located
           absolute_yaw_ics: float or list - initial conditions for yaw
           axial_induction_ics: float or list - ics for axial induction
           dt: float - time step length [s]
           yaw_rate: float - yawing speed [deg/s]
           ai_rate_limit: float - axial induction rate limit [-/s]
        """

        self.use_FLORIDyn = use_FLORIDyn
        
        if self.use_FLORIDyn:
            self.fi = wfct_floridyn.floris_interface.FlorisInterface(
                model_path
            )
            self.sim_time = 0
            self.fi.floris.farm.flow_field.mean_wind_speed = 8 # TODO: update at each time step?
            self.fi.calculate_wake() # Not sure if needed.
        else:
            # FOR DEVELOPMENT; NO LONGER SUPPORTED.
            self.fi = wfct_special.floris_interface.FlorisInterface(
                model_path
            )
            # self.fi.reinitialize_flow_field(
            #     layout_array=(self.cluster_layout.x,self.cluster_layout.y)
            # )
            #
        
        super().__init__(input_file_dir, absolute_yaw_ics, yaw_ics_range, 
            axial_induction_ics,
            dt, yaw_rate, yaw_deadband, axial_induction_rate_limit, 
            axial_induction_time_constant, 
            axial_induction_control_natural_frequency,
            axial_induction_control_damping_ratio, N=len(self.fi.layout_x))

        # Turbine specifics
        self.Cp_curve = self.fi.floris.farm.flow_field.turbine_map.turbines[0].\
            fCpInterp
        test_speeds = np.array(range(12,30))
        P = 0.5 * self.fi.floris.farm.turbines[0].air_density * \
            test_speeds**3 * np.pi * \
            self.fi.floris.farm.turbines[0].rotor_radius**2 * \
            self.Cp_curve(test_speeds)
        self.P_t_rated = P.max()
        
        self.P_turbs = self.fi.get_turbine_power()

        self.efficiency = self.fi.floris.farm.turbines[0].eta
        self.N = len(self.fi.layout_x)
        self.ai_prev = [0.33]*self.N # Initialize
        
    def step_simulator(self, ws, wd, yaw_commands, power_commands, 
                       verbose=True):
        """
        Perform steps to update control inputs, generate simulation input
        files, and run the dynamic FLORIS code to produce the output power.
        
        yaw_commands taken to be relative to oncoming winds 
        (i.e., 0 is aligned with the wind direction)

        power_commands will simply be rated in power maximizing control.
        """
    
        turbine_axial_inductions = \
            self.turbine_power_response(power_commands, ws)

        flow_axial_inductions = self.flow_response(turbine_axial_inductions)

        self.axial_induction_factors = flow_axial_inductions

        yaw_positions = self.yaw_system_response(yaw_commands, wd)

        yaw_misalignments = wd - np.array(yaw_positions)

        if self.use_FLORIDyn:
            if self.sim_time == 0:
                # Use static FLORIS to initialize FloriDyn
                self.fi.reinitialize_flow_field(
                    wind_speed=ws, 
                    wind_direction=wd,
                )
                self.fi.calculate_wake(
                    yaw_angles=yaw_misalignments, 
                    axial_induction=self.axial_induction_factors,
                )
            else:
                self.fi.reinitialize_flow_field(
                    wind_speed=ws, 
                    wind_direction=wd,
                    sim_time=self.sim_time
                )
            self.fi.calculate_wake(
                yaw_angles=yaw_misalignments, 
                axial_induction=self.axial_induction_factors,
                sim_time=self.sim_time
            )
            self.sim_time += self.dt
        else: # No need to keep track of time
            self.fi.reinitialize_flow_field(wind_speed=ws, wind_direction=wd)
            self.fi.calculate_wake(
                yaw_angles=yaw_misalignments, 
                axial_induction=self.axial_induction_factors
            )
        
        turbine_powers_raw = self.fi.get_turbine_power()

        # Apply correction to turbine_powers to account for turbine vs flow
        correction = \
            (np.array(self.turbine_axial_inductions) * \
                (1 - np.array(self.turbine_axial_inductions))**2) / \
            (np.array(self.flow_axial_inductions) * \
                (1 - np.array(self.flow_axial_inductions))**2)

        turbine_powers = correction * turbine_powers_raw
        total_power = np.sum(turbine_powers)
        
        # Record for next step
        self.P_turbs = turbine_powers

        return turbine_powers, total_power

    def visualize_flow_field(self, ws, wd, yaw_misalignments, 
                             axial_inductions):
        """
        Use FLORIS' inbuilt tools for this.
        """
        self.fi.reinitialize_flow_field(wind_speed=ws, wind_direction=wd)
        self.fi.calculate_wake(
            yaw_angles=yaw_misalignments, 
            axial_induction=axial_inductions
        )

        hor_plane = self.fi.get_hor_plane()

        return hor_plane

class SteadyStateWindSE(SimulateWindSE):
    """
    Class to simulate WindSE in steady state (no time-marching 
    behavior considered for actuators).

    NOT SUPPORTED/TESTED.
    """

    def __init__(self, input_file_dir, wd_method='rotate domain', _3d=True, 
        warp_percent=0.9):
        """
        Constructor, just calls parent.
        """
        super().__init__(input_file_dir, 
            dt=1, # Won't be used
            yaw_rate=1000, # Essentially infinite
            axial_induction_rate_limit=1000 # Essentially infinite
        )

        self.wd_method = wd_method
        self._3d = _3d
        self.warp_percent = warp_percent

# Helper functions

def bearing_to_CCW_angle(bearing):
    """
    Converts bearing angles (e.g. wind direction) to CCW angle in 
    radians (with 0 pointing West), as expected by WindSE
    """
    # Convert
    angle = (270.0 - bearing) * np.pi/180 
    
    # Wrap to (-pi, pi] (may not be needed)
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle <= -np.pi:
        angle += 2.0 * np.pi
    
    return angle

def yaw_angle_to_CCW_angle(yaw_angle):
    """
    Converts relative yaw angle (CCW positive from wind direction,
    in degrees) to a relative CCW yaw angle in radians, as expected 
    by WindSE
    """
    angle = yaw_angle * np.pi/180
    
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle <= -np.pi:
        angle += 2.0 * np.pi

    return angle

def wrap_to_360(angle_deg):
    """
    Wraps an angle (in degrees) to the interval [0, 360).
    """

    while angle_deg >= 360.0:
        angle_deg -= 360.0
    while angle_deg < 0.0:
        angle_deg += 360.0
    
    return angle_deg

def wrap_to_180(angle_deg):
    """
    Wraps an angle (in degrees) to the interval (-180, 180].
    """

    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg <= -180.0:
        angle_deg += 360.0
    
    return angle_deg

def rotate_domain(v, theta):
    """
    Map each turbine location to a coordinate system rotated by theta.
    """
    x = v[0]*np.cos(theta) + v[1]*np.sin(theta)
    y = v[1]*np.cos(theta) - v[0]*np.sin(theta)

    return x, y

def circular_mean_bearing(x):
    """
    x assumed to be a list of 1d array of bearings to be averaged.
    returns average bearing (float).
    """

    x = np.array(x)*np.pi/180

    c_avg = np.mean(np.cos(x))
    s_avg = np.mean(np.sin(x))

    x_avg = np.arctan2(s_avg, c_avg)*180/np.pi

    return wrap_to_360(x_avg)


if __name__ == "__main__":
    
    import matplotlib
    matplotlib.use('TKAgg') # Some weird workaround, maybe only for Macs/me

    # Demonstrate the code running over a small dataframe (4 entries).
    sim = SimulateWindSE('.', absolute_yaw_ics=270.0, 
                         yaw_rate=0.5, dt=4.0, 
                         wd_method='rotate domain', _3d=True)

    # Dummy dataset
    N = 30 
    wind_speeds = [6.5, 6.3, 7.1, 8.2] # m/s
    wind_directions = [270.0, 268.0, 290.7, 291.2] # degrees (absolute)
    yaw_positions = [[0.0]*N, 
                     [1.0]*N, 
                     [3.5]*12+[2.0]*(N-12),
                     [3.0]*N
                    ] # degrees (relative to oncoming wind)
    axial_induction_factors = [[0.33]*N,
                               [0.33]*8 + [0.30]*(N-8),
                               [0.30]*N,
                               [0.27]*N
                              ] # (unitless)

    df = pd.DataFrame(columns=['ws', 'wd', 'yaw', 'axial_induction'])

    for i in range(len(wind_speeds)):
        df.loc[i, 'ws'] = wind_speeds[i]
        df.loc[i, 'wd'] = wind_directions[i]
        df.loc[i, 'yaw'] = yaw_positions[i]
        df.loc[i, 'axial_induction'] = axial_induction_factors[i]

    print(df)

    # Run the code through the simulator
    df = sim.simulate_time_series(df)

    print(df)
    