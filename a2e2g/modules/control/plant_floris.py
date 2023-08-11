#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:45:06 2021

@author: svijaysh
"""





import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import floris.tools as wfct


# %% FLORIS related initialization

fi = wfct.floris_interface.FlorisInterface("data/plant/staggered_50MW.json")

# Calculate wake
fi.calculate_wake()
