import numpy as np
import pandas as pd

class BatterySimulator():
    """
    Class for simulating battery operation.
    """

    def __init__(self, capacity, charge_rate_max=np.inf, 
        discharge_rate_max=np.inf, initial_SOC=0., dt=1.0):
        """
        Constructor. 
        
        Inputs:
            capacity - Full battery capacity [J]
            charge_rate_max - maximum rate of charging [W]
            discharge_rate_max - maximum rate of discharging [W]
            initial_SOC - Starting state of charge [J]
            dt - sampling time for simulation [s]
        """

        self.P_max_in = charge_rate_max
        self.P_max_out = discharge_rate_max
        self.Q_max = capacity
        self.dt = dt
        
        # initialize the battery
        self.Q = min(initial_SOC, capacity)
    
    @staticmethod
    def MWh_to_J(Q_in_MWh):
        """
        Convert energy in MWh to energy in J.
        """
        return 1e6*(60*60)*Q_in_MWh
        
        
class SimpleSOC(BatterySimulator):
    """
    State of Charge simulator that models charging/discharing 
    inefficiencies.
    
    Inputs:
        (see parent BatterySimulator class)
        charging_efficiency - loss factor eta [-]
        discharging_efficiency - loss factor eta [-]
        storage_efficiency - loss factor to stored energy [-/s]

    efficiencies should be in range 
    (0, 1]
    """

    def __init__(self, capacity, charge_rate_max=np.inf, 
        discharge_rate_max=np.inf, charging_efficiency=1.0, 
        discharging_efficiency=1.0, storage_efficiency=1.0, 
        initial_SOC=0., dt=1.0):

        super().__init__(capacity=capacity, 
            charge_rate_max=charge_rate_max, 
            discharge_rate_max=discharge_rate_max,
            initial_SOC=initial_SOC,
            dt=dt)

        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency
        self.storage_efficiency = storage_efficiency

    def step_simulator(self, P_request=0):
        """
        Update the state of charge of the battery.

        Positive P_request/P_actual indicates chargeing; negative 
        indicates discharging

        Inputs:
            P_request - power request supplied/drawn from the battery 
                [W]
        Outputs: 
            P_actual - actual power flow to/from battery [W]
        """
        # Model storage loss
        self.Q = self.Q*(self.storage_efficiency**self.dt)

        # Determine true power flow
        if P_request > 0:
            P_request_loss = P_request*self.charging_efficiency
            P_change = min(P_request_loss, self.P_max_in, 
                (self.Q_max-self.Q)/self.dt)
            P_actual = P_change/self.charging_efficiency
        elif P_request < 0:
            P_request_loss = P_request/self.discharging_efficiency
            P_change = max(P_request_loss, -self.P_max_out, -self.Q/self.dt)
            P_actual = P_change*self.discharging_efficiency        
        else:
            P_actual = 0.
            P_change = 0.

        # Update SOC
        self.Q = self.Q + P_change*self.dt

        return P_actual




    

