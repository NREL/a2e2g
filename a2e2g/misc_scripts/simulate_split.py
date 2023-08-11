import a2e2g.modules.control_simulation.truth_sim as simulate_truth
import pandas as pd

warp_percents = [70, 80, 90]

df_control = pd.read_pickle('outputs/control_acts_2019-11-30.pkl')
yaw_rate = 1.0 # deg/s
ai_rate = 1.0 # -/s (very high, ~instantaneous)

ws_4s = df_control.ws
wd_4s = df_control.wd
yaw_cmds = df_control.yaw_cmd_base
ai_cmds = df_control.ai_cmd_base
power_4s = df_control.P_ref

# TEMP; DO TWO
n = 75 # First five minutes
ws_4s = ws_4s[:n]
wd_4s = wd_4s[:n]
yaw_cmds = yaw_cmds[:n]
ai_cmds = ai_cmds[:n]
power_4s = power_4s[:n]

df_out = pd.DataFrame({'P_ref':power_4s})
for wp in warp_percents:
    windsim = simulate_truth.SimulateWindSE(
        "a2e2g/modules/simulation", 
        absolute_yaw_ics=wd_4s[0], # Assume aligned at t=0 
        axial_induction_ics=0.33, # Assume max ai at t=0
        dt=4.0, # 4s time step
        yaw_rate=yaw_rate,
        axial_induction_rate_limit=ai_rate,
        warp_percent=wp/100
    )
    P_warp = []
    k = 0
    for agc, ws, wd, y, ai in zip(power_4s, ws_4s, wd_4s, yaw_cmds, ai_cmds):
        P_turbs, P_tot = windsim.step_simulator(
            ws, wd, y, ai, verbose=False
        )
        P_warp.append(P_tot)
        k = k+1
        print('\n---\n {:.2f}% done with '.format(k/n * 100)+str(wp)+'\n---\n')
    df_out['P_'+str(wp)] = P_warp

df_out.to_pickle('outputs/split_powers.pkl')
print(df_out)