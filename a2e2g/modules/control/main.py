#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 19:03:39 2021

@author: svijaysh
"""


# codes for control
import RL.RLcontrol as control

# modified dynamic version of floris with axial
import floris.tools as wfct
import pandas as pd

import RL.Cluster as cluster

import time
import pickle

train_controller = True
test_controller = True

# Just change the json file for different layouts
fi = wfct.floris_interface.FlorisInterface("datasets/example_input.json")
num_turbs = 30   # number of tubrines in the cluster


layout = pd.read_csv("datasets/layout.csv")
g = cluster.Cluster(layout)
layout_x, layout_y = g.getFarmCluster() 

fi.reinitialize_flow_field(layout_array=(layout_x,layout_y))


# # control object
g = control.Control(fi = fi, num_turb = num_turbs)

if train_controller: # train your controller 
    print('Training...')
    tic = time.perf_counter()
    trained_agent = g.train()
    toc = time.perf_counter()
    print('Training complete. Took {0:.2f} seconds'.format(toc - tic))

    print('Pickling agent...')
    with open('trainedagent.pkl', 'wb') as f:
        pickle.dump(trained_agent, f)
    print('Done.')

if test_controller:
    with open('trainedagent.pkl', 'rb') as f:
        trained_agent_load = pickle.load(f)

    # %%
    # # test on a different sequence
    print('Testing...')
    tic = time.perf_counter()
    g.test(trained_agent_load)
    toc = time.perf_counter()
    print('Testing complete. Took {0:.2f} seconds'.format(toc - tic))        