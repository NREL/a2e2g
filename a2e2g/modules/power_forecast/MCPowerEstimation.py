
"""
M. Sinner, 11/05/21
"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from scipy import interpolate

import a2e2g.modules.control.control_library as controllers
import a2e2g.modules.control_simulation.truth_sim as simulate_truth

class MCPowerEstimation():
    def __init__(self, use_pregenerated_sweep=True, floris_sweep_file=None, 
                 floris_json=None):
        """
        Constructor. Loads the pregenerated FLORIS data into memory.
        """
        

        if use_pregenerated_sweep:
            # If these are changed, a new sweep will need to be run.
            df_sweep_load = pd.read_csv(floris_sweep_file)
            
            self.ws_sweep = df_sweep_load['ws'].unique()
            self.wd_sweep = df_sweep_load['wd'].unique()
            
            turbine_powers = df_sweep_load.to_numpy()[:, 2:]
            self.sweep_array = turbine_powers.sum(axis=1).reshape(
                len(self.ws_sweep), len(self.wd_sweep)
            )
        else:
            print('Generating FLORIS outputs in real-time not available.')
            # TODO: We could look into this option; only needed if we want to 
            # handle a model change, such as a turbine going offline.
            # Will then need a path to the floris model.

    def predict_power(self, ws_mean, wd_mean, ws_std=None, wd_std=None, 
                      wswd_cov=None):
        """
        Predict the power output given mean conditions ws_mean, wd_mean and 
        with standard deviations ws_std and wd_std. If wswd_cov is not None, 
        ws_std and wd_std will be ignored.
        """
        if wswd_cov == None:
            wswd_cov = [[ws_std**2, 0], [0, wd_std**2]]
        
        rv = mvn([ws_mean, wd_mean], wswd_cov)

        # Get probability of each wd, ws combination in ws_sweep, wd_sweep
        p = rv.pdf([(ws, wrap_to_mean(wd, wd_mean)) 
                     for ws in self.ws_sweep for wd in self.wd_sweep])
        p = p.reshape(len(self.ws_sweep), len(self.wd_sweep))/p.sum()

        # Calculate the expected power based on that using the standard sum
        # E[f(X,Y)] = sum_x( sum_y( f(x,y)*p(x,y) ) ) 
        P_tot_mean = (self.sweep_array * p).sum() # Element-wise product
        # var(f(X,Y)) = sum_x( sum_y( f(x,y)**2 *p(x,y) ) ) - E[f(X,Y)]**2
        P_tot_var = (self.sweep_array**2 * p).sum()- P_tot_mean**2 
        P_tot_std = np.sqrt(P_tot_var)

        return P_tot_mean, P_tot_std

    def predict_power_naive(self, ws_mean, wd_mean, ws_std=None, wd_std=None, 
                            wswd_cov=None):
        """
        Predict the power output given mean conditions ws_mean, wd_mean and 
        with standard deviations ws_std and wd_std. If wswd_cov is not None, 
        ws_std and wd_std will be ignored.
        """

        if wswd_cov == None:
            wswd_cov = [[ws_std**2, 0], [0, wd_std**2]]
        
        rv = mvn([ws_mean, wd_mean], wswd_cov)

        # Get probability of each wd, ws combination in ws_sweep, wd_sweep
        p = rv.pdf([(ws, wd) 
                     for ws in self.ws_sweep for wd in self.wd_sweep])
        p = p.reshape(len(self.ws_sweep), len(self.wd_sweep))/p.sum()

        # Calculate the expected power based on that using the standard sum
        # E[f(X,Y)] = sum_x( sum_y( f(x,y)*p(x,y) ) ) 
        P_tot_mean = (self.sweep_array * p).sum() # Element-wise product
        # var(f(X,Y)) = sum_x( sum_y( f(x,y)**2 *p(x,y) ) ) - E[f(X,Y)]**2
        P_tot_var = (self.sweep_array**2 * p).sum()- P_tot_mean**2 
        P_tot_std = np.sqrt(P_tot_var)

        return P_tot_mean, P_tot_std

    def predict_power_hist(self, ws_means, wd_means, ws_stds, wd_stds, 
                           wswd_covs=None):
        """
        Make a prediction for a sequence of wd, ws. Inputs should be 
        lists with one entry for each time step.
        """

        if wswd_covs == None: # Assume uncorrelated
            wswd_covs = [[[ws_std**2, 0], [0, wd_std**2]] for \
                         ws_std, wd_std in zip(ws_stds, wd_stds)]

        P_tot_mean_hist, P_tot_std_hist = [], []
        for ws_m, wd_m, wswd_c in zip(ws_means, wd_means, wswd_covs):
            P_tot_mean, P_tot_std = self.predict_power(
                ws_mean=ws_m, wd_mean=wd_m, wswd_cov = wswd_c
            )
            P_tot_mean_hist.append(P_tot_mean)
            P_tot_std_hist.append(P_tot_std)

        return P_tot_mean_hist, P_tot_std_hist

    def center_sweep_array(self, center_index):
        """
        Center the sweep array of powers on the mean wind direction.
        Helper function to clean up main prediction function.
        """
        if round(len(self.wd_sweep)/2) == len(self.wd_sweep)/2: # even
            half_length = round(len(self.wd_sweep)/2)
        else: # odd length
            half_length = int(np.floor(len(self.wd_sweep)/2))

        shift = center_index - half_length
        original_order = [i for i in range(len(self.wd_sweep))]
        centered_order = original_order[shift:] + original_order[:shift]
        
        return centered_order

class DeterministicPowerEstimation():
    def __init__(self, use_pregenerated_sweep=True, floris_sweep_file=None,
                 floris_json=None):
        """
        Constructor. Loads the pregenerated FLORIS data into memory.
        """
        

        if use_pregenerated_sweep:
            df_sweep_load = pd.read_csv(floris_sweep_file)
            
            self.ws_sweep = df_sweep_load['ws'].unique()
            self.wd_sweep = df_sweep_load['wd'].unique()
            
            turbine_powers = df_sweep_load.to_numpy()[:, 2:]
            self.sweep_array = turbine_powers.sum(axis=1).reshape(
                len(self.ws_sweep), len(self.wd_sweep)
            )

            self.interpolated_f = interpolate.interp2d(
                self.ws_sweep, self.wd_sweep, self.sweep_array.T, kind='linear'
            )

        else:
            print('Generating FLORIS outputs in real-time not available.')
            # TODO: We could look into this option; only needed if we want to 
            # handle a model change, such as a turbine going offline.
            # Will then need a path to the floris model.

    def predict_power(self, ws, wd):
        """
        Predict the power output given mean conditions ws_mean, wd_mean and 
        with standard deviations ws_std and wd_std. If wswd_cov is not None, 
        ws_std and wd_std will be ignored.
        """
        P_tot = self.interpolated_f(ws, wd)[0]
        
        # TODO: Build in handling for wds not in range (OK for now, since 0
        # and 360 are in self.wd_sweep)
        return P_tot

    def predict_power_hist(self, ws_hist, wd_hist):
        """
        Predict the power output given mean conditions ws_mean, wd_mean and 
        with standard deviations ws_std and wd_std. If wswd_cov is not None, 
        ws_std and wd_std will be ignored.
        """
        P_tot_hist = [self.predict_power(ws, wd) for 
            ws, wd in zip(ws_hist, wd_hist)]
        
        # TODO: Build in handling for wds not in range (OK for now, since 0
        # and 360 are in self.wd_sweep)
        return P_tot_hist

def wrap_to_mean(wd, wd_m):
    """
    Wraps wd to interval (wd_m-180, wd_m+180] for circular sampling.
    """

    wd_out = wd

    while wd_out > wd_m + 180.:
        wd_out = wd_out - 360.
    while wd_out <= wd_m - 180:
        wd_out = wd_out + 360.

    return wd_out

def generate_floris_sweep(fi, wss, wds, save_file, use_wake_steering=False, 
    wake_steering_file=None):
    """
    Generate a floris sweep in the correct format.
    """
    # Make matrix
    N = len(fi.layout_x)
    I = len(wss)
    J = len(wds)
    turb_powers = np.zeros((I*J, N))

    if use_wake_steering:
        wakesteer = controllers.WakeSteering(fi, True, wake_steering_file)

    # Loop through to populate turb_powers
    for i, ws in enumerate(wss):
        for j, wd in enumerate(wds):
            fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
            if use_wake_steering:
                yaw_angles = wakesteer.step_controller(ws, wd, None)[0]
                fi.calculate_wake(yaw_angles=yaw_angles)
            else:
                fi.calculate_wake()
            turb_powers[i*J+j, :] = fi.get_turbine_power()
        print('Wind speed {0:.2f} done.'.format(ws))
    
    turb_powers[turb_powers < 0.] = 0. # Get rid of negative powers
    df = build_sweep_dataframe(turb_powers, wss, wds)
    df.to_csv(save_file, index=False)

    return turb_powers

def generate_windse_sweep(fi, wss, wds, save_file, use_wake_steering=False, 
    wake_steering_file=None):

    # Make matrix
    N = len(fi.layout_x)
    I = len(wss)
    J = len(wds)
    turb_powers = np.zeros((I*J, N))

    if use_wake_steering:
        wakesteer = controllers.WakeSteering(fi, True, wake_steering_file)
    else:
        wakesteer = controllers.Baseline(fi)

    # Initialize simulation simulator
    windsim = simulate_truth.SteadyStateWindSE(
        "./a2e2g/modules/simulation",
    )
    # Loop through to populate turb_powers
    for i, ws in enumerate(wss):
        for j, wd in enumerate(wds):
            
            fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
            yaw_angles, axial_inductions = wakesteer.step_controller(
                ws, wd, None
            )[0:2]
            turb_powers[i*J+j, :] = windsim.step_simulator(
                ws, wd, yaw_angles, axial_inductions, verbose=False
            )[0]
            print('Wind speed {0:.2f}, wind direction {1:.2f} done.'.\
                format(ws,wd))
    
    turb_powers[turb_powers < 0.] = 0. # Get rid of negative powers
    df = build_sweep_dataframe(turb_powers, wss, wds)
    df.to_csv(save_file, index=False)

    return turb_powers

def build_sweep_dataframe(turb_powers, wss, wds):
    N = turb_powers.shape[1]
    wss_vec = np.repeat(wss, len(wds)).reshape(-1,1)
    wds_vec = np.tile(wds, len(wss)).reshape(-1,1)
    data = np.concatenate((wss_vec, wds_vec, turb_powers), axis=1)
    columns = ['ws', 'wd'] + ['pow_{:03d}'.format(i) for i in range(N)]
    df = pd.DataFrame(data, columns=columns)
    return df
        
if __name__ == "__main__":
    
    Estimator = MCPowerEstimation(floris_sweep_file="FLORISfullsweep_1.csv")

    ws_m = 12.0
    ws_s = 0.1
    wd_s = 10.0
    
    P_m_n1, P_s_n = Estimator.predict_power(ws_m, 179, ws_s, wd_s)
    P_m_n2, P_s_n = Estimator.predict_power(ws_m, 181, ws_s, wd_s)
    P_m_o1, P_s_n = Estimator.predict_power_naive(ws_m, 179, ws_s, wd_s)
    P_m_o2, P_s_n = Estimator.predict_power_naive(ws_m, 181, ws_s, wd_s)
    print("Near 180:")
    print("Naive   :", abs(P_m_o1 - P_m_o2)/1e3, 'kW')
    print("Wrapped :", abs(P_m_n1 - P_m_n2)/1e3, 'kW\n')
    
    P_m_n1, P_s_n = Estimator.predict_power(ws_m, 359, ws_s, wd_s)
    P_m_n2, P_s_n = Estimator.predict_power(ws_m, 1, ws_s, wd_s)
    P_m_o1, P_s_n = Estimator.predict_power_naive(ws_m, 359, ws_s, wd_s)
    P_m_o2, P_s_n = Estimator.predict_power_naive(ws_m, 1, ws_s, wd_s)
    print("Near 0:")
    print("Naive   :", abs(P_m_o1 - P_m_o2)/1e3, 'kW')
    print("Wrapped :", abs(P_m_n1 - P_m_n2)/1e3, 'kW')


    Estimator2 = DeterministicPowerEstimation(
        floris_sweep_file="FLORISfullsweep_1.csv"
    )
    print(Estimator2.predict_power([26, ws_m+1], [179, 181])/1e3)            
