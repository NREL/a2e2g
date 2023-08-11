import datetime
import pickle

import numpy as np
import pandas as pd

from scipy.stats import norm

import pyomo.environ as pyoenv
import pyomo.opt as pyoopt

# from pyomo.environ import *
# from pyomo.opt import SolverFactory

from .Quantile_aggregate_function import quantile_aggregate

from pathlib import Path
PKGROOT = str(Path(__file__).resolve().parents[2])

class Market():
    def __init__(self, startTimeSim, stopTimeSim, market_name, bus, data_directory=None):
        self.AGC_output_filename = PKGROOT + '/grid_outputs/test_grid_data.csv'

        self.data_directory = data_directory
        self.startTimeSim = startTimeSim
        self.stopTimeSim = stopTimeSim
        self.startTimeday=datetime.datetime.strptime(startTimeSim, '%Y-%m-%d')
        self.stopTimeday=datetime.datetime.strptime(stopTimeSim, '%Y-%m-%d')
        self.market_name = market_name
        self.bus = bus
        self.wind_input = 'FLORIS' # just assuming we are always using FLORIS right now

        self.id = 0 # assuming just one day for now

        self.type_bid = 2 # start with one type
        self.quantile_adv = 0 # the lowest bid

        self.energy_scarcity_price = 9500
        self.min_penalty = 20
        self.AGC_penalty = 9500

        ramp_req_reg = 0.2
        self.AGC_frequency = 4 # in seconds
        duration = 5 * 60 / self.AGC_frequency # 5-min has 75 4-sec intervals
        wind_farm_capacity = 50 # Approximate maximum power of 30-turbine farm
        self.wind_max_ramp = wind_farm_capacity / duration
        self.max_reg_award = self.wind_max_ramp / ramp_req_reg

        self.solver = 'glpk' # solver for the pyomo optimizer

    def load_price_data(self):
        #Load real-time prices
        if self.market_name == 'ERCOT':
            startTime = self.startTimeday + datetime.timedelta(days=self.id)
            if self.data_directory is None:
                RTpriceworkbook = pd.ExcelFile(
                    PKGROOT
                    + '/data/ERCOT_data/Historical/RTMLZHBSPP_' 
                    + str((startTime).year)
                    + '.xlsx'
                )
            else:
                RTpriceworkbook = pd.ExcelFile(
                    str(Path(PKGROOT) / self.data_directory / 
                    "ERCOT_data" / "Historical" / 
                    ("RTMLZHBSPP_"+str((startTime).year)+".xlsx"))
                )
            month1 = RTpriceworkbook.sheet_names
            dfrt2 = RTpriceworkbook.parse(month1[(startTime).month - 1])       
            dfrt2 = dfrt2.loc[dfrt2["Settlement Point Name"] == self.bus] 
            dfrt2 = dfrt2.loc[
                dfrt2["Delivery Date"] == datetime.datetime.strftime(
                    startTime, format='%m/%d/%Y'
                )
            ]
            dfrt2 = dfrt2.rename(columns={'Settlement Point Price': 'RT Energy Price'})
            dfrt2 = dfrt2.rename(
                columns={"Delivery Interval":"Delivery Price Interval"}
            )
            dfrt2AS = dfrt2.loc[
                :, ("Delivery Date", "Delivery Hour", "Delivery Price Interval")
            ]
            dfrt2AS["RT Regup price"] = 2100 # ARBITRARY ASSUMPTION
            dfrt2AS["RT Regdown price"] = 2000 # ARBITRARY ASSUMPTION

        #Load DA ASprices
        if self.market_name == 'ERCOT':
            startTime = self.startTimeday+datetime.timedelta(days=self.id)
            if self.data_directory is None:
                df_reg = pd.read_csv(
                    PKGROOT
                    + '/data/ERCOT_data/Historical/DAMASMCPC_'
                    + str((startTime).year)
                    + '.csv'
                )
            else: 
                df_reg = pd.read_csv(
                    str(Path(PKGROOT) / self.data_directory / 
                    "ERCOT_data" / "Historical" / 
                    ("DAMASMCPC_"+str((startTime).year)+".csv"))
                )  
            df_reg = df_reg.loc[
                df_reg["Delivery Date"] == datetime.datetime.strftime(
                    startTime, format='%-m/%-d/%Y' # MS: CHANGED 11/10/21
                )
            ]
            df_reg.index = range(0, len(df_reg))
            for i in range(0, len(df_reg)):
                df_reg.at[i, "Hour Ending"] = float(
                    df_reg.at[i, "Hour Ending"][0:df_reg.at[i, "Hour Ending"].find(":")]
                )
            

        # Read day-ahead energy prices
        if self.market_name == 'ERCOT':
            startTime = self.startTimeday + datetime.timedelta(days=self.id)
            if self.data_directory is None:
                priceworkbook = pd.ExcelFile(
                    PKGROOT
                    + '/data/ERCOT_data/Historical/DAMLZHBSPP_'
                    + str((startTime).year)
                    + '.xlsx'
                )
            else:
                priceworkbook = pd.ExcelFile(
                    str(Path(PKGROOT) / self.data_directory / 
                    "ERCOT_data" / "Historical" / 
                    ("DAMLZHBSPP_"+str((startTime).year)+".xlsx"))
                )
            month1 = priceworkbook.sheet_names
            df = priceworkbook.parse(month1[(startTime).month - 1])       
            df = df.loc[
                (
                    df['Delivery Date'] == datetime.datetime.strftime(
                        startTime, format='%m/%d/%Y'
                    )
                )
                & (df['Settlement Point'] == self.bus) 
            ]
            df.index=range(0, len(df))
            for i in range(0, len(df)):
                df.at[i,"Hour Ending"] = float(
                    df.at[i, "Hour Ending"][0:df.at[i, "Hour Ending"].find(":")]
                )
        DAprices = pd.merge(df, df_reg, on='Hour Ending')
        if self.market_name == 'ERCOT':
            RTprices=pd.merge(
                dfrt2,
                dfrt2AS,
                on=["Delivery Hour", "Delivery Price Interval"]
            )

        # RTprices = RTprices.rename(columns={'RegUPService':'RT Regup price'})
        # RTprices = RTprices.rename(columns={'RegDNService':'RT Regdown price'})

        return dfrt2, dfrt2AS, DAprices, RTprices


    def day_ahead_bidding(self, dfrt2, dfrt2AS, df_grid_inputs=None):
        if df_grid_inputs is None:
            print('Loading pre-computed FLORIS data...')
            df_grid_inputs = pickle.load(
                open(
                    '../floris/floris_outputs_2019_11_30.p', 'rb'
                )
            )
            # time_array = pd.date_range(
            #     start='2019-11-30', end='2019-12-01', periods=None, freq='5T'
            # )[:-1]

            # df_grid_inputs = pd.DataFrame()
            # df_grid_inputs["Time"] = time_array
            # df_grid_inputs["PWR_STD"] = prob_power[0]
            # df_grid_inputs["PWR_MEAN"] = prob_power[1]

        # block 1: quantities for DAbid [fixed quantiles to bid]
        DAwindforecast=pd.DataFrame()
        DAwindforecast["quantile"] = [0.09, 0.5]
        DAwindforecast["quantile low"] = [0.365, 0.5]
        DAwindforecast["quantile high"] = [0.006, 0.5]

        quantile_adv = 0 # the lowest bid

        if (self.wind_input == 'FLORIS') & (self.type_bid != 1):
            # df_grid_inputs has the forecast    
            df_grid_inputs["Delivery Hour"] = df_grid_inputs["Time"].dt.hour
            df_grid_inputs["Delivery Hour"] = df_grid_inputs["Delivery Hour"] + 1
            for j in range(0, len(DAwindforecast)):
                df_grid_inputs["quantile " + str(DAwindforecast.at[j,"quantile"])] = (
                    norm.ppf(DAwindforecast.at[j, "quantile"])
                    * df_grid_inputs["PWR_STD"]
                    + df_grid_inputs["PWR_MEAN"]
                )
            for i in range(0,24): 
                for j in range(0, len(DAwindforecast)):
                # TODO: Assumption alert: Conservative assumption for intertemporal
                # correlation -- we can update it later
                    if len(
                        df_grid_inputs["PWR_STD"].loc[
                            (df_grid_inputs["Delivery Hour"] == i + 1)
                            & (round(df_grid_inputs["PWR_STD"], 2) > 0.5)
                        ]
                    ) > 6:
                        DAwindforecast.at[j, i + 1] = quantile_aggregate(
                            df_grid_inputs["PWR_MEAN"].loc[
                                (df_grid_inputs["Delivery Hour"] == i + 1)
                                & (round(df_grid_inputs["PWR_STD"], 2) > 0.5)
                            ].values.tolist(),
                            #
                            df_grid_inputs["PWR_STD"].loc[
                                (df_grid_inputs["Delivery Hour"] == i + 1)
                                & (round(df_grid_inputs["PWR_STD"], 2) > 0.5)
                            ].values.tolist(),
                            #
                            DAwindforecast.at[j, "quantile"],
                            #
                            df_grid_inputs[
                                "quantile "
                                + str(DAwindforecast.at[j,"quantile"])
                            ].loc[df_grid_inputs["Delivery Hour"] == i + 1].mean()
                        )
                    else:
                        DAwindforecast.at[j, i + 1] = np.quantile(
                            df_grid_inputs["PWR_MEAN"].loc[
                                (df_grid_inputs["Delivery Hour"] == i + 1)
                            ],
                            #
                            DAwindforecast.at[j, "quantile"]
                        )
                    #the normal can go into negative domain - which is not reasonable
                    if DAwindforecast.at[j, i + 1] == -1000:
                        DAwindforecast.at[j, i + 1] = np.quantile(
                            df_grid_inputs["PWR_MEAN"].loc[
                                (df_grid_inputs["Delivery Hour"] == i + 1)
                            ],
                            #
                            DAwindforecast.at[j, "quantile"]
                        )
                    DAwindforecast.at[j, i + 1] = max(0, DAwindforecast.at[j, i + 1])

        #block 2: prices at which those fixed quantiles of MW will be offered
        dfnew2 = pd.DataFrame()   
        dfnew2["RT quantile"] = DAwindforecast["quantile"]
        if self.type_bid != 6:
            dfnew2["RT quantile low"] = dfnew2["RT quantile"]
            dfnew2["RT quantile high"] = dfnew2["RT quantile"]
        else:
            dfnew2["RT quantile low"] = DAwindforecast["quantile low"]
            dfnew2["RT quantile high"] = DAwindforecast["quantile high"]
        if (self.type_bid <= 2) or (self.type_bid == 6):
        #assumption 1 for RT prices: perfectly known       
            for hour in range(1, 25):
                if self.market_name == "ERCOT":
                    a = dfrt2["RT Energy Price"].loc[
                        dfrt2["Delivery Hour"] == hour
                    ].mean()
                    b = max(
                        self.min_penalty, dfrt2["RT Energy Price"].loc[
                            dfrt2["Delivery Hour"] == hour
                        ].mean()
                    )
                    c = b
                elif self.market_name == "SPP":
                    a = dfrt2["RT Energy Price"].loc[
                        dfrt2["Delivery Hour"] == hour
                    ].mean()
                    b = dfrt2AS["RegUPService"].loc[
                        dfrt2AS["Delivery Hour"] == hour
                    ].mean()
                    c = dfrt2AS["RegDNService"].loc[
                        dfrt2AS["Delivery Hour"] == hour
                    ].mean()
                if (DAwindforecast.at[quantile_adv, hour] < self.max_reg_award):
                        dfnew2["Energy" + str(hour)] = a
                        dfnew2["Regup" + str(hour)] = (
                            b * dfnew2["RT quantile low"] + a 
                            - (1 - dfnew2["RT quantile low"]) * 0.25
                            * dfrt2["RT Energy Price"].loc[
                                dfrt2["Delivery Hour"] == hour
                            ].mean()
                        )
                        dfnew2["Regdn" + str(hour)] = (
                            c * dfnew2["RT quantile low"]
                            + (1 - dfnew2["RT quantile low"]) * 0.25
                            * dfrt2["RT Energy Price"].loc[
                                dfrt2["Delivery Hour"] == hour
                            ].mean()
                        )
                elif DAwindforecast.at[quantile_adv,hour] >= self.max_reg_award:
                        dfnew2["Energy" +str(hour)] = a
                        dfnew2["Regup" +str(hour)] = (
                            b * dfnew2["RT quantile high"] + a
                            - (1 - dfnew2["RT quantile high"]) * 0.25
                            * dfrt2["RT Energy Price"].loc[
                                dfrt2["Delivery Hour"] == hour
                            ].mean()
                        )
                        dfnew2["Regdn" +str(hour)] = (
                            c * dfnew2["RT quantile high"]
                            + (1 - dfnew2["RT quantile high"]) * 0.25
                            * dfrt2["RT Energy Price"].loc[
                                dfrt2["Delivery Hour"] == hour
                            ].mean()
                        )
        #assumption 2 :price taker in DA market
        elif self.type_bid == 3:
            for hour in range(1, 25):
                dfnew2[hour] = 0
                dfnew2[hour].loc[dfnew2["RT quantile"] <= 0.5] = 0
                dfnew2[hour].loc[dfnew2["RT quantile"] > 0.5] = (
                    self.energy_scarcity_price
                )
        #assumption 3 :worst case scenario
        elif self.type_bid == 4:
            for hour in range(1, 25):
                dfnew2[hour] = self.energy_scarcity_price * dfnew2["RT quantile"]    

        DAbid = pd.DataFrame()

        for product in ["Energy","Regup","Regdn"]:
            for hour in range(1, 25):
                listperhour = []
                for q in range(0, len(DAwindforecast)):                 
                    listperhour.append((
                        DAwindforecast[hour].iloc[q],
                        dfnew2[str(product) + str(hour)].iloc[q]
                    ))
                DAbid[str(product) + " " + str(hour)] = listperhour

        return DAbid


    def day_ahead_simulations(self, DAprices, DAbid):
        """
        Created on Mon Sep 14 23:27:23 2020

        @author: espyrou

        This piece of code uses as input:
        DAprices from Step3pre_Load_price_data.py
        DAbid from step3b

        Needs pyomo and solvers

        Provides output in df_DA_result
        Can be updates to solve all 24 hours at once, not important for wind when no
        intertemporal constraint

        """
        DAprices = DAprices.sort_values(by='Hour Ending')
        df_DA_result = pd.DataFrame(
            DAprices[["Settlement Point Price", "REGUP ", "REGDN"]]
        )
        df_DA_result = df_DA_result.rename(
            columns={
                "Settlement Point Price": "DA Energy price",
                "REGUP ":"DA Regup price",
                "REGDN": "DA Regdown price"
            }
        )

        # Decide on day-ahead awards based on prices and forecast wind generation
        for j in range(1, 25):
            model = pyoenv.ConcreteModel()
            sizemod = range(0, len(DAbid))
            model.y = pyoenv.Var(sizemod, domain = pyoenv.NonNegativeReals)
            model.z = pyoenv.Var(sizemod, domain = pyoenv.NonNegativeReals)
            model.p = pyoenv.Var(sizemod, domain = pyoenv.NonNegativeReals)
            
            model.Objective = pyoenv.Objective(
                expr = sum(
                    (
                        (
                            df_DA_result.at[j - 1, "DA Energy price"]
                            - DAbid["Energy "+str(j)].iloc[i][1]
                        ) * model.y[i]
                    ) for i in range(0, len(DAbid))
                )
                + sum(
                    (
                        df_DA_result.at[j - 1, "DA Regup price"]
                        * model.z[i]
                        - DAbid["Regup " + str(j)].iloc[i][1] 
                        * model.z[i]
                    ) for i in range(0, len(DAbid))
                )
                + sum(
                    (
                        df_DA_result.at[j - 1, "DA Regdown price"]
                        - DAbid["Regdn " + str(j)].iloc[i][1]
                    ) * model.p[i] for i in range(0, len(DAbid))
                ),
                sense = pyoenv.maximize
            )
            
            model.constraints = pyoenv.ConstraintList() 
            # The following sets of constraints makes sure that the cleared amount is
            # in line with the submitted bid
            model.Constraint1 = pyoenv.Constraint(
                expr = model.y[0] <= DAbid["Energy " + str(j)].iloc[0][0]
            )
            model.Constraint2 = pyoenv.Constraint(
                expr = model.z[0] <= DAbid["Regup " + str(j)].iloc[0][0]
            )
            model.Constraint3 = pyoenv.Constraint(
                expr = model.p[0] <= DAbid["Regdn " + str(j)].iloc[0][0]
            )
            for i in range(1, len(DAbid)):
                model.constraints.add(
                    model.y[i] <= DAbid["Energy " + str(j)].iloc[i][0]
                    - DAbid["Energy " + str(j)].iloc[i - 1][0]
                )
                model.constraints.add(
                    model.z[i] <= DAbid["Regup " + str(j)].iloc[i][0]
                    - DAbid["Regup " + str(j)].iloc[i - 1][0]
                )
                model.constraints.add(
                    model.p[i] <= DAbid["Regdn " + str(j)].iloc[i][0]
                    - DAbid["Regdn " + str(j)].iloc[i - 1][0]
                )
            #Additional technical constraints
            #Reg down less than energy award
            #reg down lower or equal to energy

            model.Constraint4 = pyoenv.Constraint(
                expr = sum(
                    model.p[i] for i in range(0, len(DAbid))
                ) <= sum(
                    model.y[i] for i in range(0, len(DAbid)))
                )
            #energy and reg up lower or equal to capacity
            for i in range(1, len(DAbid)):
                model.constraints.add(
                    model.y[i] + model.z[i] <=
                    DAbid["Energy " + str(j)].iloc[i][0]
                    - DAbid["Energy " + str(j)].iloc[i - 1][0])
            model.Constraint5 = pyoenv.Constraint(
                expr = sum(
                    (model.y[i] + model.z[i]) for i in range(0, 1)
                ) <= DAbid["Energy " + str(j)].iloc[0][0])

            # compliance-performance related constrained: approximation as if the
            # participant had the best algorithm to decide on how to split the
            # offer                               
            model.Constraint6 = pyoenv.Constraint(
                expr = sum(
                    (model.z[i] + model.p[i]) for i in range(0, len(DAbid))
                ) <= DAbid["Energy " + str(j)].iloc[self.quantile_adv][0])    
            model.Constraint7 = pyoenv.Constraint(
                expr = sum(
                    model.z[i] for i in range(0,len(DAbid))
                ) <= self.max_reg_award
            )
            model.Constraint8 = pyoenv.Constraint(
                expr = sum(
                    model.p[i] for i in range(0,len(DAbid))
                ) <= self.max_reg_award
            )
            opt = pyoopt.SolverFactory(self.solver)

            # TODO: results not used below; is this important?    
            results = opt.solve(model)
            df_DA_result.at[j - 1, "DA Energy award"] = sum(
                model.y[i].value for i in range(0, len(DAbid))
            )
            df_DA_result.at[j - 1, "DA Regup award"] = sum(
                model.z[i].value for i in range(0, len(DAbid))
            )
            df_DA_result.at[j - 1, "DA Regdown award"] = sum(
                model.p[i].value for i in range(0, len(DAbid))
            )
                
        df_DA_result["DA Energy_revenue"] = df_DA_result["DA Energy award"].multiply(
                df_DA_result["DA Energy price"],
                axis = 0
            )
        df_DA_result["DA Regup_revenue"] = df_DA_result["DA Regup award"].multiply(
                df_DA_result["DA Regup price"],
                axis = 0
            )
        df_DA_result["DA Regdn_revenue"] = df_DA_result["DA Regdown award"].multiply(
                df_DA_result["DA Regdown price"],
                axis = 0
            )
        df_DA_result["Delivery Hour"]=df_DA_result.index + 1

        return df_DA_result

    def real_time_bidding_advisor(self, df_DA_result, df_grid_inputs=None):
        # TODO: figure out if this is actually needed
        actual_region = 'SYSTEM_WIDE'
        if df_grid_inputs is None:
            print('Loading pre-computed FLORIS data...')
            df_grid_inputs = pickle.load(open(
                '../floris/floris_outputs_2019_11_30.p', 'rb'
            ))

        df_grid_inputs["Delivery Hour"] = df_grid_inputs["Time"].dt.hour
        df_grid_inputs["Delivery Hour"] = df_grid_inputs["Delivery Hour"] + 1

        df_RT_result = pd.DataFrame()
        RTwindforecast = pd.DataFrame()
        RTwindforecast["quantile"] = [0.5] # simple assumption for the moment since we only have deterministic real-time forecasts
        if (self.wind_input == 'ERCOT') or (self.wind_input == 'SPP'):   
            df_grid_inputs = df_grid_inputs.rename(
                columns={"INTERVAL_ENDING":"Time", actual_region: "PWR_MEAN"}
            )
            df_grid_inputs["PWR_STD"] = 0 
            df_grid_inputs["PWR_MEAN"].loc[df_grid_inputs["PWR_MEAN"] < 0] = 0.0        
        df_RT_result["Delivery Hour"] = df_grid_inputs["Delivery Hour"]
        df_RT_result["Delivery Interval"] = pd.to_datetime(
            df_grid_inputs["Time"], format="%m/%d/%Y %H:%M"
        ).dt.minute / (5) + 1
        #df_RT_result["Delivery Interval"].loc[df_RT_result["Delivery Interval"]==0]=12

        for j in range(0, len(RTwindforecast)):
            df_RT_result["quantile" + str(RTwindforecast.at[j, "quantile"])] = (
                norm.ppf(1 - RTwindforecast.at[j, "quantile"]) 
                * df_grid_inputs["PWR_STD"] + df_grid_inputs["PWR_MEAN"]
            )            

        df_RT_result = pd.merge(
            df_RT_result,
            df_DA_result[
                [
                    "Delivery Hour", "DA Energy award", "DA Regup award",
                    "DA Regdown award", "DA Energy price", "DA Regup price",
                    "DA Regdown price"
                ]
            ], on='Delivery Hour'
        )
        for product in ["Energy", "Regup", "Regdown"]:
        # note that this does not cover only excess but deficit as well since the
        # generator might have to buy back
            for j in range(0, len(RTwindforecast)):
                df_RT_result["Excess " + product + str(j)] = (
                    df_RT_result["quantile" + str(RTwindforecast.at[j, "quantile"])]
                    - df_RT_result["DA " + product + " award"]
                )
                if (self.market_name == 'ERCOT') & (product != 'Energy'):
                    df_RT_result["Excess " + product + str(j)].loc[
                        df_RT_result["Excess " + product + str(j)] > 0
                    ] = 0

        RTbid = pd.DataFrame()
        for product in ["Energy", "Regup", "Regdown"]:
            for i in range(0, len(df_RT_result)):
                listperhour = []
                for q in range(0, len(RTwindforecast)):           
                    listperhour.append(
                        (df_RT_result.at[i, "Excess " + product + str(q)], 0)
                    ) #TODO: hardcoded alert: here I assume the excess will cost us 0!!
                RTbid[str(product) + " " + str(i)] = listperhour
        
        return RTbid, df_RT_result

    def real_time_market_simulation(self, df_RT_result, RTbid, RTprices):
        if self.market_name=='ERCOT':
            df_RT_result["Delivery Price Interval"] = (
                df_RT_result["Delivery Interval"] / 3
            )
            df_RT_result["Delivery Price Interval"] = np.ceil(
                df_RT_result["Delivery Price Interval"]
            )
            df_RT_result = pd.merge(
                df_RT_result, RTprices, on=["Delivery Hour","Delivery Price Interval"]
            )
        elif self.market_namearket == 'SPP':
            df_RT_result = pd.merge(
                df_RT_result, RTprices, on=["Delivery Hour","Delivery Interval"]
            )

        #What-if simulator based on Real-Time historical prices
        for j in range(0, len(df_RT_result)):
                model2 = pyoenv.ConcreteModel()
                sizemod = range(0, len(RTbid))
                model2.y = pyoenv.Var(sizemod, domain=pyoenv.Reals)
                model2.z = pyoenv.Var(sizemod, domain=pyoenv.Reals)
                model2.p = pyoenv.Var(sizemod, domain=pyoenv.Reals)       
                model2.Objective = pyoenv.Objective(
                    expr = sum(
                        (
                            (
                                df_RT_result.at[j, "RT Energy Price"]
                                - RTbid["Energy " + str(j)].iloc[i][1]
                            ) * model2.y[i]
                        ) for i in range(0, len(RTbid))
                    )
                    + sum(
                        (
                            df_RT_result.at[j, "RT Regup price"]
                            * model2.z[i] - RTbid["Regup "+ str(j)].iloc[i][1]
                            * model2.z[i]
                        ) for i in range(0, len(RTbid))
                    )
                    + sum(
                        (
                            df_RT_result.at[j, "RT Regdown price"]
                            - RTbid["Regdown " + str(j)].iloc[i][1]
                        ) * model2.p[i] for i in range(0, len(RTbid))
                    ),
                    sense = pyoenv.maximize
                )
                model2.constraints = pyoenv.ConstraintList() 
                # The following sets of constraints makes sure that the cleared amount
                # is in line with the submitted bid
                model2.Constraint1 = pyoenv.Constraint(
                    expr = model2.y[0] <= RTbid["Energy " + str(j)].iloc[0][0]
                )
                model2.Constraint2 = pyoenv.Constraint(
                    expr = model2.z[0] <= RTbid["Regup " + str(j)].iloc[0][0]
                )
                model2.Constraint3 = pyoenv.Constraint(
                    expr = model2.p[0] <= RTbid["Regdown " + str(j)].iloc[0][0]
                )
                for i in range(1, len(RTbid)):
                    model2.constraints.add(
                        model2.y[i] <= RTbid["Energy " + str(j)].iloc[i][0]
                        - RTbid["Energy " + str(j)].iloc[i - 1][0]
                    )
                    model2.constraints.add(
                        model2.z[i] <= RTbid["Regup " + str(j)].iloc[i][0]
                        - RTbid["Regup " + str(j)].iloc[i - 1][0]
                    )
                    model2.constraints.add(
                        model2.p[i] <= RTbid["Regdown " + str(j)].iloc[i][0]
                        - RTbid["Regdown " + str(j)].iloc[i - 1][0]
                    )
                # Additional technical constraints
                # Reg down less than energy award
                # reg down lower or equal to energy
                model2.Constraint4 = pyoenv.Constraint(
                    expr = sum(model2.p[i] for i in range(0, len(RTbid)))
                        <= sum(model2.y[i] for i in range(0, len(RTbid)))
                        + df_RT_result.at[j, "DA Energy award"]
                    )
                # energy and reg up lower or equal to capacity
                model2.Constraint5 = pyoenv.Constraint(
                    expr = df_RT_result.at[j, "DA Energy award"]
                    + df_RT_result.at[j, "DA Regup award"]
                    + sum((model2.y[i] + model2.z[i]) for i in range(0, len(RTbid)))
                    <= df_RT_result.at[j, "quantile0.5"]
                )
                # compliance-performance related constrained: approximation as if the
                # participant had the best algorithm to decide on how to split the
                # offer                               
                # model2.Constraint6    
                model2.Constraint7 = pyoenv.Constraint(
                    expr = sum(model2.z[i] for i in range(0, len(RTbid)))
                    <= self.max_reg_award
                )
                model2.Constraint8 = pyoenv.Constraint(
                    expr = sum(model2.p[i] for i in range(0, len(RTbid)))
                    <= self.max_reg_award
                )
                # given that we want to consider buy back- an additional constraint is
                # needed that it can not buy back more than the day-ahead award
                model2.Constraint9 = pyoenv.Constraint(
                    expr = sum(model2.z[i] for i in range(0, len(RTbid)))
                    >= -df_RT_result.at[j, "DA Regup award"]
                )
                model2.Constraint10 = pyoenv.Constraint(
                    expr = sum(model2.p[i] for i in range(0, len(RTbid)))
                    >= -df_RT_result.at[j, "DA Regdown award"]
                )
                model2.Constraint11 = pyoenv.Constraint(
                    expr = sum(model2.y[i] for i in range(0, len(RTbid)))
                    >= -df_RT_result.at[j, "DA Energy award"]
                )
                
                opt = pyoenv.SolverFactory(self.solver)
                results2 = opt.solve(model2)
                df_RT_result.at[j, "RT Energy award"] = sum(
                    model2.y[i].value for i in range(0, len(RTbid))
                )
                df_RT_result.at[j, "RT Regup award"] = sum(
                    model2.z[i].value for i in range(0, len(RTbid))
                )
                df_RT_result.at[j,"RT Regdown award"] = sum(
                    model2.p[i].value for i in range(0, len(RTbid))
                )

        df_RT_result["RT Energy_revenue"] = df_RT_result["RT Energy award"].multiply(
                df_RT_result["RT Energy Price"],
                axis=0
            )

        if self.market_name != 'ERCOT':
            df_RT_result["RT Regup_revenue"] = df_RT_result["RT Regup award"].multiply(
                    df_RT_result["RT Regup price"],
                    axis=0
                )
            df_RT_result["RT Regdn_revenue"] = df_RT_result["RT Regdown award"].multiply(
                    df_RT_result["RT Regdown price"],
                    axis=0
                )
            
        else:
            df_RT_result["RT Regup_revenue"] = 0     
            df_RT_result["RT Regdn_revenue"] = 0

        return df_RT_result

    def create_system_regulation_signal(self, create_AGC=False):
        startTime=datetime.datetime.strftime(
            self.startTimeday + datetime.timedelta(days=self.id), format='%Y-%m-%d'
        )
        if (self.market_name == 'SPP') & (create_AGC is True):
            ACE = pd.read_csv(PKGROOT + "/SPP_data/1minACE_SPP.csv")
            ACE["GMTTime"] = ACE["GMTTime"].str.replace("T", " ")
            ACE["GMTTime"] = ACE["GMTTime"].str.replace("Z", "")
            ACE["date"] = pd.to_datetime(ACE["GMTTime"], format="%Y-%m-%d %H:%M:%S")
            
            ACEfocus=ACE.loc[ACE["date"].dt.strftime('%Y-%m-%d') == startTime]
            
            if len(ACEfocus) > 0:
                ACEfocus["Up"] = 0
                ACEfocus["Down"] = 0
                ACEfocus["Up"].loc[ACEfocus["1 Min ACE Average"] < 0] = 1
                ACEfocus["Down"].loc[ACEfocus["1 Min ACE Average"] > 0] = 1
                template = pd.read_csv(PKGROOT + "/SPP_data/template_SPP_AGCsignal.csv")
                ACEfocus = ACEfocus.sort_values(['date'], ascending=[True])
                ACEfocus["sec"] = (
                    ACEfocus["date"].dt.hour * 3600
                    + ACEfocus["date"].dt.minute * 60+ACEfocus["date"].dt.second
                )
                x = range(0, 3600 * 24, 4)
                xp = ACEfocus["sec"]
                fp = ACEfocus["Up"].astype(float)
                AGC = pd.DataFrame()
                AGC["reg up"] = np.interp(x, xp, fp)
                # TODO: changed 'temp' to 'ACEfocus' below; make sure is correct
                fp = ACEfocus["Down"].astype(float)
                AGC["reg dn"] = np.interp(x, xp, fp)
                template[template.columns[len(template.columns) - 2]] = AGC["reg up"]
                template[template.columns[len(template.columns) - 1]] = -AGC["reg dn"]
                template.to_csv(
                    PKGROOT + "/SPP_data/Synthetic_data/" + self.market_name
                    + "_syntheticAGCsignal" + startTime.replace("-", "_") + ".csv",
                    index=False
                )
                AGC = template
            else:
                # read a dummy signal if data for this day misisng 
                AGC = pd.read_csv(
                    self.market_name + "_data/Synthetic_data/"
                    + self.market_name + "_dummyAGCsignal.csv"
                )
        elif (
            (self.market_name == 'ERCOT')
            & (datetime.datetime.strptime(startTime, '%Y-%m-%d').year == 2019)
            & (create_AGC == True)
        ):
            # USING A SHORTENED VERSION OF FULL DATA FOR DEMONSTRATION.
            freq = pd.read_csv(
                #str(Path(PKGROOT) / self.data_directory / "ERCOT_data" / "ERCOT_TrueTime_BT_Hz_2s_2019.csv")
                str(Path(PKGROOT) / self.data_directory / "ERCOT_data" / "ERCOT_TrueTime_BT_Hz_4s_2019-09-28.csv")
            )
            freq["date"] = pd.to_datetime(
                freq[freq.columns[0]].str[0:19], format="%Y-%m-%d %H:%M:%S"
            )
            freqfocus = freq.loc[
                freq["date"].dt.strftime('%Y-%m-%d') == startTime
            ]
            # this next line was used for offline generation of AGC signal in August
            # TODO: 2nd line was originally uncommented; is it right to use the 1st?
            # freqfocus = freq.loc[freq["date"].dt.month == 11]
            freqfocus = freqfocus.loc[freqfocus["date"].dt.second % 4 == 0]
            if len(freqfocus) > 0:
                freqfocus["freq dev"] = (freqfocus["HZ1"] + freqfocus["HZ2"]) / 2 - 60
                # simple assumption
                freqfocus["Up"] = 0
                freqfocus["Down"] = 0
                freqfocus["Up"].loc[freqfocus["freq dev"] < 0] = 1
                freqfocus["Down"].loc[freqfocus["freq dev"] > 0] = 1
                # more elaborate assumption based on exhaustiooon rate -needs a month
                # of data
                # exhaust=pd.read_excel('Auxiliary/Reg_exhaustion_rate_Oct2019.xlsx')
                # freqfocus["Up"]=0
                # freqfocus["Down"]=0
                # for hour in range(0,24):
                # temp=freqfocus.loc[freqfocus["date"].dt.hour==hour]
                # tempup=temp.loc[temp["freq dev"]<0]
                # tempdn=temp.loc[temp["freq dev"]>0]
                # denup=np.quantile(tempup["freq dev"],exhaust.at[hour,'regup'])
                # dendn=np.quantile(tempdn["freq dev"],1-exhaust.at[hour,'regdown'])
                #
                # freqfocus["Up"].loc[
                #     (freqfocus["freq dev"]<0) & (freqfocus["date"].dt.hour == hour)
                # ] = freqfocus["freq dev"].loc[
                #     (freqfocus["freq dev"] < 0) & (freqfocus["date"].dt.hour == hour)
                # ] / denup
                # freqfocus["Down"].loc[
                #     (freqfocus["freq dev"]>0)
                #     & (freqfocus["date"].dt.hour == hour)
                # ] = freqfocus["freq dev"].loc[
                #     (freqfocus["freq dev"] > 0)
                #     & (freqfocus["date"].dt.hour == hour)
                # ] / dendn
                # freqfocus["Up"]=np.minimum(freqfocus["Up"],1)
                # freqfocus["Down"]=np.minimum(freqfocus["Down"],1)
                template = pd.read_csv(
                    #PKGROOT + "/data/ERCOT_data/template_ERCOT_AGCsignal.csv"
                    str(Path(PKGROOT) / self.data_directory / "ERCOT_data" / "template_ERCOT_AGCsignal.csv")
                )
                # next lines useful for running a month off-line
                # days=freqfocus["date"].dt.day.unique()
                # for i in range(0,len(days)):
                #     daystr=freqfocus["date"].loc[freqfocus["date"].dt.day==days[i]].dt.strftime('%Y-%m-%d').drop_duplicates()
                #     temp=freqfocus.loc[freqfocus["date"].dt.day==days[i]]
                #     temp.index=range(0,len(temp))
                #     template[template.columns[len(template.columns)-2]]=temp["Up"]
                #     template[template.columns[len(template.columns)-1]]=-temp["Down"]
                #     template.to_csv("ERCOT_data/Synthetic_data/"+market+"_syntheticAGCsignal_elaborate_"+daystr.str.replace("-","_").values[0]+".csv", index=False)     
                freqfocus.index = range(0, len(freqfocus))
                template[template.columns[len(template.columns) - 2]] = freqfocus["Up"]
                template[template.columns[len(template.columns) - 1]] = -freqfocus["Down"]
                template.to_csv(
                    str(Path(PKGROOT) / self.data_directory / "ERCOT_data" / 
                    (self.market_name
                    + "_syntheticAGCsignal_" + startTime.replace("-", "_") + ".csv")),
                    index=False
                )

                AGC = template
        else:
            # read AGC previously created
            # TODO: this data isn't on the repo; need to generate above
            # AGC = pd.read_csv(
            #     PKGROOT + '/data/' + self.market_name + "_data/Synthetic_data/"
            #     + self.market_name + "_syntheticAGCsignal_elaborate_"
            #     + startTime.replace("-", "_") + ".csv"
            # )
            # AGC = pd.read_csv(
            #     PKGROOT + '/data/' + self.market_name + "_data/Synthetic_data/"
            #     + self.market_name + "_syntheticAGCsignal_"
            #     + startTime.replace("-", "_") + ".csv"
            # )
            AGC = pd.read_csv(
                str(Path(PKGROOT) / self.data_directory / "ERCOT_data" / 
                (self.market_name
                + "_syntheticAGCsignal_" + startTime.replace("-", "_") + ".csv")),
                index=False
            ) # MS 11/10/21
            # Note that for ERCOT the code is called "Read_json_with_reg_deployment.py"
            # and creates AGC for multiple days
        
        return AGC

    def create_wind_plant_regulation_signal(self, AGC, df_RT_result):
        AGC = pd.merge(
            AGC, df_RT_result[[
                'Delivery Hour', 'Delivery Interval', 'RT Energy award',
                'RT Regup award', 'RT Regdown award', 'DA Energy award',
                'DA Regup award', 'DA Regdown award'
            ]], on=['Delivery Hour', 'Delivery Interval']
        )
        x = range(0, int(24 * 3600 / self.AGC_frequency), 1)
        xp = range(0, int(24 * 3600 / self.AGC_frequency), int(
            24 * 3600 / (self.AGC_frequency * len(df_RT_result))
        ))
        fp = df_RT_result["DA Energy award"] + df_RT_result["RT Energy award"]
        fp[0] = (
            df_RT_result.at[0, "DA Energy award"]
            + df_RT_result.at[0, "RT Energy award"]
        )
        AGC["Ideal energy"] = np.interp(x, xp, fp)
        # Forward contractual promise: be able to linearly ramp between energy awards
        # and provide reg up and reg down at appropriate intervals
        if self.market_name == 'ERCOT':
            AGC["Total Regdn"] = AGC["DA Regdown award"]
            AGC["Total Regup"] = AGC["DA Regup award"]
        else:       
            AGC["Total Regdn"] = AGC["DA Regdown award"] + AGC["RT Regdown award"]
            AGC["Total Regup"] = AGC["DA Regup award"] + AGC["RT Regup award"]
        AGC["Forward Capacity"] = (
            AGC[["Ideal energy", "Total Regdn"]].max(axis=1) + AGC['Total Regup']
        )
        # Assume a pro-rata distribution (note that depending on how the AGC signal is
        # created it might not be pro-rata but always calling on wind)  
        AGC["Ideal Regup"] = AGC["Total Regup"].multiply(
            AGC["normalized regup"], axis=0
        )
        AGC["Ideal Regdn"] = AGC["Total Regdn"].multiply(
            AGC["normalized regdn"],axis=0
        )

        AGC["Ideal signal"] = (
            AGC[["Ideal energy","Total Regdn"]].max(axis=1)
            + AGC["Ideal Regup"]
            + AGC["Ideal Regdn"]
        )

        sizemod = range(0, len(AGC))
        # calculate the AGC signal that should be able to achieve based on the forward
        # RT contractual promise
        model3 = pyoenv.ConcreteModel()
        model3.x = pyoenv.Var(sizemod, domain = pyoenv.NonNegativeReals)
        model3.y = pyoenv.Var(sizemod, domain = pyoenv.NonNegativeReals)
        model3.Objective = pyoenv.Objective(
                expr = sum(self.AGC_penalty * model3.y[i] for i in range(1, len(AGC))),
                sense = pyoenv.minimize
            )
        model3.constraints = pyoenv.ConstraintList() 
        for i in range(0, len(AGC)):
                model3.constraints.add(model3.x[i] <= AGC.at[i, "Forward Capacity"])
                model3.constraints.add(
                    model3.y[i] >= AGC.at[i, "Ideal signal"] - model3.x[i]
                )
                model3.constraints.add(
                    model3.y[i] >= model3.x[i] - AGC.at[i, "Ideal signal"]
                )
        for i in range(1, len(AGC)):
                model3.constraints.add(
                    model3.x[i] - model3.x[i-1] <= self.wind_max_ramp
                )
                model3.constraints.add(
                    model3.x[i-1] - model3.x[i] <= self.wind_max_ramp
                )
        opt = pyoenv.SolverFactory(self.solver)
        results3 = opt.solve(model3)

        for i in range(0,len(AGC)):
            AGC.at[i, "Basepoint signal"] = model3.x[i].value

        #AGC[['timestamp','Basepoint signal']].to_csv(
        #    self.AGC_output_filename, index=False
        #)

        return AGC


if __name__ == "__main__":
    startTimeSim = "2019-11-30"
    stopTimeSim = "2019-12-01"

    market_name = "ERCOT"
    # all options for ERCOT ["HB_SOUTH","HB_NORTH","HB_WEST", "HB_BUSAVG"]
    bus="HB_BUSAVG"

    market = Market(startTimeSim, stopTimeSim, market_name, bus)

    # step 3a
    dfrt2, dfrt2AS, DAprices, RTprices = market.load_price_data()

    # step 3b
    DAbid = market.day_ahead_bidding(dfrt2, dfrt2AS, df_grid_inputs=None)

    # step 3c
    df_DA_result = market.day_ahead_simulations(DAprices, DAbid)

