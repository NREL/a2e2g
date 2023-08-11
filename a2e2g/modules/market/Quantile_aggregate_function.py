# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:54:08 2021

@author: espyrou

This is a simple way of calculating the "quantile" we should submit in a particular hour 
when we have sub-hourly power forecasts with mean in Z and std in Y 
and the forecasts follow a normal distribution

Basically function F calculates the percentile a value X corresponds to for each sub-hourly interval i
Then in a, all percentiles are added up and function F becomes 0 when the average value of sub-hourly percentiles matches the target percentile W

P is an initial estimate of the value that will solve the root finding problem.
"""

import scipy.optimize
from scipy.stats import norm


def quantile_aggregate(Z,Y,W,P):  

    def F(X):
        a=0
        for i in range(0,len(Z)):
            a=    norm.cdf(X,round(Z[i],2),round(Y[i],2))+a       
        return a-W*len(Z)
    
    # import scipy.optimize
    try:
        x = scipy.optimize.broyden1(F,P, f_tol=1e-1)
        return x
    except Exception:
        return -1000