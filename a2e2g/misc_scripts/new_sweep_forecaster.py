import numpy as np
import pandas as pd

import a2e2g.modules.power_forecast.MCPowerEstimation as mcpe
import a2e2g.modules.control.RL.Cluster as cluster

import matplotlib
matplotlib.use('TKAgg') # Some weird workaround, maybe only for Macs/me

from floris import tools as wfct

wind_plant = 'staggered_50MW'

fi = wfct.floris_interface.FlorisInterface(
    "./a2e2g/data_local/"+wind_plant+".json"
)

# Has 360 doubled still. Careful of that. 
mcpe.generate_floris_sweep(fi, np.linspace(0,25,60), np.linspace(0,360,100), 
    "./a2e2g/data_local/floris_datasets/FLORISfullsweep_"+wind_plant+".csv",
    use_wake_steering=False
)