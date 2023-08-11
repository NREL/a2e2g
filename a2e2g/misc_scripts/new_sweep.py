import numpy as np
import pandas as pd

import a2e2g.modules.power_forecast.MCPowerEstimation as mcpe
import a2e2g.modules.control.RL.Cluster as cluster

import matplotlib
matplotlib.use('TKAgg') # Some weird workaround, maybe only for Macs/me

from floris import tools as wfct

floris_sweep = False
windse_sweep = True

floris_json = None # Include floris model json file here.

fi = wfct.floris_interface.FlorisInterface(floris_json)

layout = pd.read_csv("./a2e2g/modules/control/datasets/layout.csv")
h = cluster.Cluster(layout)
layout_x, layout_y = h.getFarmCluster() 
fi.reinitialize_flow_field(layout_array=(layout_x,layout_y))

# Has 360 doubled still. Careful of that. 
if floris_sweep:
    mcpe.generate_floris_sweep(fi, np.linspace(0,25,60), np.linspace(0,360,50), 
        './a2e2g/data_local/floris_datasets/FLORISfullsweep_4_wakesteer.csv',
        use_wake_steering=True,
        wake_steering_file="./a2e2g/modules/control/wake_steering_offsets.pkl"
    )

if windse_sweep:
    wss = np.linspace(20,24,3)
    mcpe.generate_windse_sweep(fi, wss, np.linspace(0,360,37), 
        './a2e2g/data_local/floris_datasets/WindSEfullsweep_{0}.csv'.\
            format(wss[0]),
        use_wake_steering=True,
        wake_steering_file="./a2e2g/modules/control/wake_steering_offsets.pkl"
    )