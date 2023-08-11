import pandas as pd
import a2e2g.modules.control_simulation.truth_sim as simulator
import time

# Load up the controller data
data_directory = './a2e2g/data_local/control/'
controller = 'a2e2g'
mod = 'test'

df_cont = pd.read_pickle(data_directory+'Cont_'+controller+mod+\
    '_2019-09-28.pkl')

df_cont.rename(
    columns={'yaw_cmd_'+controller:'yaw', 
             'ai_cmd_'+controller:'axial_induction'},
    inplace=True
)

yaw_rate = 1.0 # deg/s
ai_rate = 1.0 # -/s (very high, ~instantaneous)

windsim = simulator.SimulateFLORIS(
    "a2e2g/modules/simulation", 
    absolute_yaw_ics=df_cont.wd.iloc[0], # Assume aligned at t=0 
    axial_induction_ics=0.33, # Assume max ai at t=0
    dt=4.0, # 4s time step
    yaw_rate=yaw_rate,
    axial_induction_rate_limit=ai_rate,
    use_FLORIDyn=False
) # 90% split still seems good?

t_start = time.perf_counter()
df_sim = windsim.simulate_time_series(df_cont, verbose=True)
t_end = time.perf_counter()
print(df_sim.P_true)
df_sim.to_pickle('outputs/Sim_FLORIS_'+controller+mod+'_2019-09-28.pkl')
print('Done. Took {0:.2f} minutes to complete {1} steps.'.\
    format((t_end-t_start)/60, len(df_sim)))
