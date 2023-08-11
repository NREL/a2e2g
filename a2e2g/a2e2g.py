# main module for a2e2g wrappers



# Is this called at every time pointor do you want a whole trajectroy as the output

import pickle
from matplotlib import use
import pandas as pd
import numpy as np
import datetime
import multiprocessing

from pathlib import Path
PKGROOT = str(Path(__file__).parent.resolve())
running_on_eagle = False

import a2e2g.modules.forecast.forecast_backend as anen
# import a2e2g.modules.floris.DayAheadPowerEstimation as dape
import a2e2g.modules.power_forecast.MCPowerEstimation as mcpe
# import a2e2g.modules.floris.FiveMinuteAheadPowerEstimation as fmape

#import a2e2g.modules.control.RL.RLcontrol as control
import a2e2g.modules.control.RL.Cluster as cluster
import a2e2g.modules.control.control_library as controllers
import a2e2g.modules.control.RL.timeseries as ts
#import a2e2g.modules.control.floris.tools as wfct
from floris import tools as wfct
from floris.utilities import wrap_360

import a2e2g.modules.control_simulation.truth_sim as simulate_truth
import a2e2g.modules.control_simulation.truth_sim_battery as simulate_battery

class a2e2g():
    def __init__(self, 
        data_directory, 
        wind_plant='staggered_50MW'):
        self.data_directory = data_directory
        self.wind_plant = wind_plant
    
    def forecast(self, utc_minus, tz_current, tz_adjust, wind_data_file, hrrr_files, day='2019-11-30', integration=False): 
        if integration is False:
            tz_adjust_hires = tz_adjust * 12

            wind_obs = anen.obs_load(filename=wind_data_file)

            hrrr_wspd, hrrr_wdir, hrrr_ti, obs_wspd, obs_wdir, obs_ti = anen.hrrr_obs_process(
                hrrr_files=hrrr_files, 
                obs_file=wind_obs,
                utc_minus = utc_minus,
                tz_current = tz_current,
                tz_adjust = tz_adjust,
                tz_adjust_hires = tz_adjust_hires,
                day=day
            )

            final_wspd, final_wdir, final_ti = anen.analog_forecast(
                total_hrrr_wspd = hrrr_wspd, 
                total_hrrr_wdir = hrrr_wdir, 
                total_hrrr_ti = hrrr_ti, 
                total_obs_wspd = obs_wspd, 
                total_obs_wdir = obs_wdir, 
                total_obs_ti = obs_ti,
                day=day)
        else:
            final_wspd, final_wdir, final_ti = pickle.load(
                open(
                    PKGROOT + '/modules/forecast/forecast_outputs.p', 'rb'
                )
            )
        # MS 11/11/20 Handing for 0 standard deviation values
        threshold_stde = 0.1
        final_wspd['Analog Stde'][final_wspd['Analog Stde'] <= threshold_stde] = threshold_stde
        final_wdir['Analog Stde'][final_wdir['Analog Stde'] <= threshold_stde] = threshold_stde

        forecast_outputs = [final_wspd, final_wdir, final_ti]
        return forecast_outputs

    def corrected_floris_estimation(self, forecast_inputs, integration=False):
        # not enough data for day ahead yet, will follow with Andrew for 
        # atleast a months worth of data
        if integration is False:
            prob_power = dape.probabilisticPowerEstimationDayAhead(data_directory=self.data_directory, forecast=forecast_inputs)
            # TODO: setup to match df_grid_inputs for Elina's code; need to get time
            # from Andrew's code
            print(forecast_inputs[0].index)
            print(prob_power)
            prob_power["Time"] = forecast_inputs[0].index
            floris_outputs = prob_power
        else:
            floris_outputs = pickle.load(
                open(
                    PKGROOT + '/modules/floris/floris_outputs_2019_11_30.p', 'rb'
                )
            )
            # print(forecast_inputs[0].index)
            # floris_outputs["Time"] = forecast_inputs[0].index
            # time_array = pd.date_range(
            #     start='2019-11-30', end='2019-12-01', periods=None, freq='5T'
            # )#[0:3]
            # time_array = pd.to_datetime(time_array.values, format="%m/%d/%Y %H:%M")

        
        return floris_outputs

    def floris_estimation(self, forecast, wake_steering=False, scale_to_MW=False):
        """
        Similar to above, but no machine-learned correction.
        """
        # Sweeps can be generated using the new_sweep_forecaster.py script
        # in supporting_scripts
        if wake_steering:
            data_file = "FLORISfullsweep_"+self.wind_plant+"_wakesteer.csv"
        else:
            data_file = "FLORISfullsweep_"+self.wind_plant+".csv"

        estimator = mcpe.MCPowerEstimation(
            floris_sweep_file=str(Path(PKGROOT) / self.data_directory / "plant" / data_file)
        )

        P_mean_est, P_stddev_est = estimator.predict_power_hist(
            forecast[0]['Analog Mean'].to_list(),
            forecast[1]['Analog Mean'].to_list(),
            forecast[0]['Analog Stde'].to_list(),
            forecast[1]['Analog Stde'].to_list()
        )

        # Convert to Dataframe format
        if scale_to_MW:
            floris_outputs = pd.DataFrame(
                {'Time':forecast[0].index,
                'PWR_STD':np.array(P_stddev_est)/1e6,
                'PWR_MEAN':np.array(P_mean_est)/1e6}
            )
        else:
            floris_outputs = pd.DataFrame(
                {'Time':forecast[0].index,
                'PWR_STD':P_stddev_est,
                'PWR_MEAN':P_mean_est}
            )

        return floris_outputs

    def load_price_data(self, market):
        # step 3a
        dfrt2, dfrt2AS, DAprices, RTprices = market.load_price_data()
        return dfrt2, dfrt2AS, DAprices, RTprices

    def day_ahead_bidding(self, market, dfrt2, dfrt2AS, floris_inputs):      
        # step 3b
        day_ahead_bid = market.day_ahead_bidding(
            dfrt2, dfrt2AS, df_grid_inputs=floris_inputs
        )
        return day_ahead_bid

    def intermediate_bidding(self, market, day_ahead_bid, day_ahead_prices):
        # step 3c
        intermediate_bid = market.day_ahead_simulations(day_ahead_prices, day_ahead_bid)
        return intermediate_bid

    def floris_short_term(self, wind_speed, wind_direction, TI, integration=False):
        if integration is False:
            prob_power = fmape.probabilisticPowerEstimation5min()
            # TODO: setup to match df_grid_inputs for Elina's code; need to get time
            # from Andrew's code
        else:
            short_term_power_estimate = pickle.load(
                open(
                    PKGROOT + '/modules/floris/floris_outputs_2019_11_30.p', 'rb'
                )
            )

            # time_array = pd.date_range(
            #     start='2019-11-30', end='2019-12-01', periods=None, freq='5T'
            # )#[0:3]
            # time_array = pd.to_datetime(time_array.values, format="%m/%d/%Y %H:%M")

        # floris_outputs = (prob_power, time_array)
        return short_term_power_estimate

    def floris_deterministic(self, short_term_forecast, wake_steering=False, scale_to_MW=False):

        if wake_steering:
            data_file = "FLORISfullsweep_"+self.wind_plant+"_wakesteer.csv"
        else:
            data_file = "FLORISfullsweep_"+self.wind_plant+".csv"

        estimator = mcpe.DeterministicPowerEstimation(
            floris_sweep_file=str(Path(PKGROOT) / self.data_directory / "plant" / data_file)
        )
        
        P_est = estimator.predict_power_hist(
            short_term_forecast['WS'].to_list(),
            short_term_forecast['WD'].to_list()
        )

        # Convert to Dataframe format
        if scale_to_MW:
            floris_outputs = pd.DataFrame(
                {'Time':short_term_forecast['Time'],
                'PWR_STD':[0.]*len(P_est),
                'PWR_MEAN':np.array(P_est)/1e6}
            )
        else:
            floris_outputs = pd.DataFrame(
                {'Time':short_term_forecast['Time'],
                'PWR_STD':[0.]*len(P_est),
                'PWR_MEAN':P_est}
            )

        return floris_outputs

    def real_time_AGC_signal(self, market, df_DA_result, short_term_power_estimate):
        RTbid, df_RT_result = market.real_time_bidding_advisor(
            df_DA_result, df_grid_inputs=short_term_power_estimate
        )
        return RTbid, df_RT_result

    def load_test_winds(self, dir, daterange=None):
        df_wind = pd.read_csv(dir).rename(columns={'Unnamed: 0':'datetime'})
        df_wind.datetime = pd.to_datetime(df_wind.datetime)

        if daterange != None:
            df_wind = df_wind[
                (df_wind.datetime >= pd.to_datetime(daterange[0])) & 
                (df_wind.datetime < pd.to_datetime(daterange[1]) + \
                                    pd.Timedelta(1, 'days'))]
        
        #df_wind.interpolate(method='polynomial', order=2, inplace=True)
        df_wind.fillna(method='ffill', inplace=True)
        df_wind.fillna(method='bfill', inplace=True)

        return df_wind
    
    def load_short_term_forecast(self, dir, daterange=None):
        df_temp = pd.read_csv(dir).rename(columns={'Unnamed: 0':'Time'})
        df_temp.Time = pd.to_datetime(df_temp.Time)

        if daterange != None:
            df_temp = df_temp[
                (df_temp.Time > pd.to_datetime(daterange[0])) & 
                (df_temp.Time <= pd.to_datetime(daterange[1]) + \
                    pd.Timedelta(1, 'days'))]
            # 00:00 time stamp missing in data!
            df_temp = pd.concat([df_temp.iloc[0:1], df_temp], 
                ignore_index=True, axis=0)
            df_temp.iloc[0,0] = df_temp.iloc[1].Time - \
                pd.Timedelta(5, 'minutes')        
        
        df_temp.fillna(method='ffill', inplace=True)
        df_temp.fillna(method='bfill', inplace=True)

        # Build desired dataframe
        df_st_forecast = pd.DataFrame({'Time':df_temp.Time})
        df_st_forecast['WS'] = np.linalg.norm(
            df_temp[['75m_U', '75m_V']].to_numpy(), axis=1)
        df_st_forecast['WD'] = df_temp['75m_WD']

        return df_st_forecast

    def short_term_persistence(self, wind_data_file, daterange=None):

        df_1min = pd.read_csv(wind_data_file).rename(columns={'Unnamed: 0':'Time'})
        df_1min.Time = pd.to_datetime(df_1min.Time)
        
        # Average over 5 minutes (assigns to start)
        df_5min = df_1min.set_index('Time').resample('5T').mean().reset_index()

        # Forecast 5 minutes ahead using persistence (shift by 10 minutes 
        # due to assignment to start)
        df_5min.Time = df_5min.Time + pd.Timedelta(value=10, unit='minutes')

        if daterange != None:
            df_5min = df_5min[
                (df_5min.Time > pd.to_datetime(daterange[0])) & 
                (df_5min.Time <= pd.to_datetime(daterange[1]) + \
                    pd.Timedelta(1, 'days'))]
            # 00:00 time stamp missing in data from Andrew!            
            #df_temp = pd.concat([df_temp.iloc[0:1], df_temp], 
            #    ignore_index=True, axis=0)
            #df_temp.iloc[0,0] = df_temp.iloc[1].Time - \
            #    pd.Timedelta(5, 'minutes')        
        
        df_5min.fillna(method='ffill', inplace=True)
        df_5min.fillna(method='bfill', inplace=True)

        # Build desired dataframe
        df_st_forecast = pd.DataFrame({'Time':df_5min.Time})
        df_st_forecast['WS'] = np.linalg.norm(
            df_5min[['75m_U', '75m_V']].to_numpy(), axis=1)
        df_st_forecast['WD_temp'] = df_5min['75m_WD']
        df_st_forecast['WD'] = wrap_360(270. - np.rad2deg(
            np.arctan2(df_5min['75m_V'], df_5min['75m_U'])))

        return df_st_forecast

    def RL_controller(self, df_wind=None, df_AGC_signal=None):
        
        # Get floris model up and running
        fi = wfct.floris_interface.FlorisInterface(floris_json)
        layout = pd.read_csv("./a2e2g/modules/control/datasets/layout.csv")
        h = cluster.Cluster(layout)
        layout_x, layout_y = h.getFarmCluster() 
        fi.reinitialize_flow_field(layout_array=(layout_x,layout_y))

        # Extract useful data. 
        g = control.Control(fi=fi, num_turb=len(layout_x))
        wind_4s, power_4s, time_pow_4s = g.data(
            df_wind=df_wind, 
            df_AGC_signal=df_AGC_signal
        )
        
        # with open("/projects/a2e2g/datashare/control/trainedagent.pkl", "rb") as f:
        #     trained_agent = pickle.load(f)
        with open("./a2e2g/trainedagent2.pkl", "rb") as f:
            trained_agent = pickle.load(f)

        yaw = []
        axial_induction = []
        print('Generating controls...')
        for agc, ws in zip(power_4s, wind_4s):
            y, a = g.getControlInput(agc, [ws]*g.num_turb, trained_agent)
            yaw.append(y)
            axial_induction.append(a)

        return yaw, axial_induction, wind_4s.tolist()

    def WindSE(self, yaw, axial_induction, wind_speed):
        windsim = simulate_truth.SimulateWindSE(PKGROOT+'/modules/simulation')
        
        # Put data into dataframe for use in simulate_time_series
        wind_direction = [0]*len(yaw) # CHECK---no wind direction specified?
        df_sim = pd.DataFrame({'ws': wind_speed,
                               'wd': wind_direction,
                               'yaw': yaw,
                               'axial_induction': axial_induction
                               })

        # Run simulation, return dataframe with powers column
        df_sim = windsim.simulate_time_series(df_sim)
        
        # Extract powers, convert to list to return.
        true_power_output = df_sim.Ptot_true.tolist()
        return true_power_output

    def simulate_operation_RL(self, df_wind=None, df_AGC_signal=None):

        # Establish controller
        if running_on_eagle:
            fi = wfct.floris_interface.FlorisInterface(floris_json)
            layout = pd.read_csv(
                PKGROOT + "/data/floris_datasets/layout.csv"
            )
            with open(PKGROOT + "/control_data/trainedagent.pkl", "rb") as f:
                trained_agent = pickle.load(f)
        else:
            fi = wfct.floris_interface.FlorisInterface(floris_json)
            layout = pd.read_csv(
                "./a2e2g/modules/control/datasets/layout.csv"
            )
            with open("./a2e2g/modules/control/trainedagent.pkl", "rb") as f:
                trained_agent = pickle.load(f)

        h = cluster.Cluster(layout)
        layout_x, layout_y = h.getFarmCluster() 
        fi.reinitialize_flow_field(layout_array=(layout_x,layout_y))
        g = control.Control(fi=fi, num_turb=len(layout_x))
        
        # Establish WindSE simulator
        windsim = simulate_truth.SimulateWindSE(
            PKGROOT+"/modules/simulation", yaw_rate=100
        )
        
        wd = 270. # NOT CHANGING!
        axial_induction = [0.33]*g.num_turb # Fixed in Floris.
        yaw = [0]*g.num_turb # Initial values
        
        actual_powers = []
        for agc, ws in zip(power_4s, wind_4s):
            
            P_turbs, P_tot = windsim.step_simulator(
                ws, wd, yaw, axial_induction, False
            ) # TODO: W_turbs should be an output
            W_turbs = [ws]*g.num_turb # TODO: remove this line
            actual_powers.append(P_tot)

            yaw, a = g.getControlInput(agc, W_turbs, trained_agent)

        return actual_powers, power_4s

    def simulate_operation(self, 
        control_cases=['base'], 
        use_battery=False, 
        df_wind=None, 
        df_AGC_signal=None, 
        delayed_control=False, 
        closed_loop_simulator=None, 
        sim_range=None, 
        dt=4.0, 
        nondefault_batteries=None, 
        name_extensions=None, 
        parallelize_over_cases=False):

        """
        If nondefault_batteries is not None, it should be a list containing 
        BatterySimulator or None for control cases where use_battery is False.
        """

        # Prepare input data
        time_power_dt, ws_dt, wd_dt, power_dt, = ts.GetData_WSWD(
            df_wind=df_wind,
            df_AGC_signal=df_AGC_signal,
            dt=dt
        )
        dt_agc = 4
        if sim_range is not None:
            if type(sim_range[0]) is datetime.datetime:
                # Reduce data to simulation range of interest
                n_start = round(df_AGC_signal[df_AGC_signal.time==\
                    sim_range[0]].index[0] * dt_agc/dt)
                n_end = round(df_AGC_signal[df_AGC_signal.time==sim_range[1]].\
                    index[0] * dt_agc/dt)
                time_power_dt = pd.date_range(sim_range[0], sim_range[1], 
                    freq=str(dt)+'S')[:-1]
            elif type(sim_range[0]) is int:
                # Manually reduce data using indices
                n_start = sim_range[0]#4275
                n_end = sim_range[1]#6525
                time_power_dt = time_power_dt[n_start:n_end]
            else:
                raise TypeError('Invalid type for sim_range.')
            ws_dt = ws_dt[n_start:n_end]
            wd_dt = wd_dt[n_start:n_end]
            power_dt = power_dt[n_start:n_end]


        # Initialize dataframe for storing outputs
        df = pd.DataFrame({
            'time':time_power_dt,
            'ws':ws_dt, 'wd':wd_dt,
            'P_ref':[agc*1e6 for agc in power_dt]
        })

        # Parse use_battery:
        num_cases = len(control_cases)
        if type(use_battery) is bool:
            use_battery = [use_battery]*num_cases
        elif len(use_battery) != num_cases:
            raise ValueError('Please specificy use_battery as list of'+\
                ' same length as control_cases.')
        if nondefault_batteries is None:
            nondefault_batteries = [None]*num_cases
        elif len(nondefault_batteries) != num_cases:
            raise ValueError('Please specificy nondefault_batteries as list'+\
                ' of same length as control_cases (pad with Nones).')
        if name_extensions is None:
            name_extensions = ['']*num_cases
        elif len(name_extensions) != num_cases:
            raise ValueError('Please specificy name_extensions as list'+\
                ' of same length as control_cases.')


        # To make parallelization simpler
        simulation_parameters = {
            "closed_loop_simulator":closed_loop_simulator,
            "wind_plant":self.wind_plant,
            "delayed_control":delayed_control,
            "running_on_eagle":running_on_eagle,
            "dt":dt,
            "df":df
        } 

        # With single case function defined at end of script, loop over the cases.
        if parallelize_over_cases:
            with multiprocessing.Pool(processes=num_cases) as pool:
                df_cases = pool.starmap(single_case, 
                                        zip(control_cases, 
                                            use_battery, 
                                            nondefault_batteries,
                                            name_extensions,
                                            [simulation_parameters]*num_cases
                                            )
                                        )
        else:
            df_cases = map(single_case, 
                           control_cases, 
                           use_battery, 
                           nondefault_batteries,
                           name_extensions,
                           [simulation_parameters]*num_cases
                           )
        df = pd.concat([df, *df_cases], axis='columns')

        return df

    def compute_value(self, updated_AGC_signal, true_power_output):
        value_of_wind_farm = []
        return value_of_wind_farm

# Function for running a single case. 
def single_case(control_case, use_batt, battsim, ext, sim_params): 
    # Settings for truth model
    yaw_rate = 0.3 # deg/s
    yaw_deadband = 8.0
    ai_rate = 1.0 # -/s (very high, ~instantaneous---use linear dynamics)
    ai_tau = 12.0
    ai_omega_n = 2*np.pi/10
    ai_zeta = 0.6

    df = sim_params["df"]

    # TODO: Set up FLORIS option here.
    np.random.seed(0)
    absolute_yaw_ics = df.wd.values[0] #+ \
    yaw_ics_range = 0 # ICs sampled from a uniform distribution
        #np.random.uniform(low=-5.0, high=5.0, size=30) # <--- TODO: HARDCODED!
    if sim_params["closed_loop_simulator"] == 'WindSE':
        windsim = simulate_truth.SimulateWindSE(
            PKGROOT+"/modules/simulation", 
            absolute_yaw_ics=absolute_yaw_ics,
            yaw_ics_range=yaw_ics_range, # Random noise added to IC
            axial_induction_ics=0.33, # Assume max ai at t=0
            dt=sim_params["dt"], # 4s time step
            yaw_rate=yaw_rate,
            yaw_deadband=yaw_deadband,
            axial_induction_rate_limit=ai_rate,
            axial_induction_time_constant=ai_tau
        )
    elif sim_params["closed_loop_simulator"] == 'FLORIS':
        windsim = simulate_truth.SimulateFLORIS(
            PKGROOT+"/modules/simulation", 
            PKGROOT+"/data_local/"+sim_params["wind_plant"]+".json",
            absolute_yaw_ics=absolute_yaw_ics, # Assume aligned at t=0 
            yaw_ics_range=yaw_ics_range, # Random noise added to IC
            axial_induction_ics=0.33, # Assume max ai at t=0
            dt=sim_params["dt"], # 4s time step
            yaw_rate=yaw_rate,
            yaw_deadband=yaw_deadband,
            axial_induction_rate_limit=ai_rate,
            axial_induction_time_constant=ai_tau,
            axial_induction_control_natural_frequency=ai_omega_n,
            axial_induction_control_damping_ratio=ai_zeta
        )
    elif sim_params["closed_loop_simulator"] == 'FLORIDyn':
        windsim = simulate_truth.SimulateFLORIS(
            str(Path(PKGROOT) / "modules" / "simulation"), 
            str(Path(PKGROOT) / "data" / "plant" / 
                (sim_params["wind_plant"]+".json")),
            absolute_yaw_ics=absolute_yaw_ics, # Assume aligned at t=0 
            yaw_ics_range=yaw_ics_range, # Random noise added to IC
            axial_induction_ics=0.33, # Assume max ai at t=0
            dt=sim_params["dt"], # 4s time step
            yaw_rate=yaw_rate,
            yaw_deadband=yaw_deadband,
            axial_induction_rate_limit=ai_rate,
            axial_induction_time_constant=ai_tau,
            axial_induction_control_natural_frequency=ai_omega_n,
            axial_induction_control_damping_ratio=ai_zeta,
            use_FLORIDyn=True
        )
    else:
        windsim = None

    # Establish battery simulator, if desired
    if use_batt:
        if battsim is None: # Use default settings below
            battery_capacity=simulate_battery.\
                BatterySimulator.MWh_to_J(5*4) # 5MW over 4 hours
            battsim = simulate_battery.SimpleSOC(
                capacity=battery_capacity,
                charge_rate_max=5e6,
                discharge_rate_max=5e6,
                charging_efficiency=0.92,
                discharging_efficiency=0.92,
                storage_efficiency=(1-1e-5),
                initial_SOC=battery_capacity/2,
                dt=sim_params["dt"]
            )
        else:
            battery_capacity = battsim.Q_max

    # Establish controller
    if sim_params["running_on_eagle"]:
        fi = wfct.floris_interface.FlorisInterface(
            str(Path(PKGROOT) / "data" / "plant" / 
                (sim_params["wind_plant"]+".json"))
        )
    else:
        fi = wfct.floris_interface.FlorisInterface(
            str(Path(PKGROOT) / "data" / "plant" / 
                (sim_params["wind_plant"]+".json"))
        )
        lookup_file = str(Path(PKGROOT) / "data" / "plant" /
            "wake_steering_offsets.pkl")
    # h = cluster.Cluster(layout)
    # layout_x, layout_y = h.getFarmCluster() 
    # fi.reinitialize_flow_field(layout_array=(layout_x,layout_y))
    if control_case in ['base', 'base_b']:
        controller = controllers.Baseline(fi)
    elif control_case == 'wake_steering':
        controller = controllers.WakeSteering(fi, True, lookup_file)
    elif control_case == 'a2e2g':
        controller = controllers.ActivePowerControl(fi, lookup_file)
    elif control_case in ['PI', 'PI_b']:
        controller = controllers.PIPowerControl(fi, dt=sim_params["dt"], 
            use_battery=use_batt)
    elif control_case in ['P', 'P_b']:
        controller = controllers.PIPowerControl(fi, dt=sim_params["dt"], 
            K_p_reduction=0.1, K_i_reduction=0, use_battery=use_batt)
    elif control_case in ['P_ws', 'P_ws_b']:
        controller = controllers.PIPowerControl(fi, dt=sim_params["dt"], 
            K_p_reduction=0.1, K_i_reduction=0, use_battery=use_batt, 
            wake_steering_parameters={
                "use_lookup_table":True,
                "lookup_file":lookup_file,
                "activation_delay":300
            })
    elif control_case in ['ind_apc', 'ind_apc_b']:
        controller = controllers.IndividualAPC(fi, dt=sim_params["dt"])
    else:
        raise ValueError('Invalid control case!')

    if sim_params["delayed_control"]: # Initialize controller outputs (to base)
        y_next = [0]*controller.N
        ai_next = [0.33]*controller.N
            
    actual_powers = []
    power_commands = []
    axial_inductions_achieved = []
    yaw_commands = []
    yaw_positions_absolute = []
    estimated_powers = []
    turbine_powers = []
    farm_powers = []
    battery_SOCs = []
    k = 0 # Time step counter
    P_turbs = [500e3]*controller.N # Assume start at 500kW
    P_cmds = P_turbs # Assume start at 500kW
    for agc, ws, wd in zip(df.P_ref.values/1e6, df.ws.values, df.wd.values): 
        if sim_params["delayed_control"]:
            # Apply control from ws[k], wd[k], power_4s[k] at time 
            # k+1.
            y_cmds = y_next 
            P_cmds = P_next
            y_next, P_next, P_est = controller.step_controller(ws, wd, agc)
        else:
            if use_batt:
                y_cmds, P_cmds, P_est = controller.step_controller(
                    ws, wd, agc, 
                    P_turbs=P_turbs,
                    battery_SOC=battsim.Q,
                    battery_capacity=battery_capacity,
                    battery_rate_limit=battsim.P_max_in
                )
            else:
                y_cmds, P_cmds, P_est = controller.step_controller(
                    ws, wd, agc, 
                    P_turbs=P_turbs
                )

        
        if windsim != None:
            P_turbs, P_farm = windsim.step_simulator(
                ws, wd, y_cmds, P_cmds, False
            )
            turbine_powers.append(P_turbs)
            if use_batt:
                P_batt = battsim.step_simulator(P_farm-agc*1e6)
                P_out = P_farm - P_batt
                battery_SOCs.append(battsim.Q)
            else:
                P_out = P_farm
            actual_powers.append(P_out)
            farm_powers.append(P_farm)
        else:
            P_turbs = controller.P_turbs_est
        power_commands.append(P_cmds)
        yaw_commands.append(y_cmds)
        estimated_powers.append(P_est)
        axial_inductions_achieved.append(windsim.axial_induction_factors)
        yaw_positions_absolute.append(windsim.absolute_yaw_positions)

        k = k+1
        if sim_params["closed_loop_simulator"] == 'WindSE':
            print_res = 1
        else:
            print_res = 100
        if k % print_res == 0:
            print('---\n {0:.2f}% through '.format(k/len(df)*100.)+
                    control_case+ext+'\n---')

    df_case = pd.DataFrame(data={
        'P_est_'+control_case+ext:estimated_powers,
        'P_cmd_'+control_case+ext:power_commands,
        'yaw_cmd_'+control_case+ext:yaw_commands,
        'ai_'+control_case+ext:axial_inductions_achieved,
        'yaw_abs_'+control_case+ext:yaw_positions_absolute
        }, index=df.index
    )

    if windsim != None:
        df_case['P_act_'+control_case+ext] = actual_powers
        df_case['P_farm_'+control_case+ext] = farm_powers
        df_case['P_turbs_'+control_case+ext] = turbine_powers
    if use_batt:
        df_case['E_batt_'+control_case+ext] = battery_SOCs
    else:
        df_case['P_batt_'+control_case+ext] = [0]*len(df)

    return df_case