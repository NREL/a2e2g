
from .ddpg_torch import Agent
import numpy as np

from . import timeseries as ts
from . import environment_ac as env

from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt


class Control:
    def __init__(self, fi, num_turb = 9):
        '''
        For later: these are not likely to change but add as inputs if 
        required
        
        These are currently set up for 3X3 but should be changed to test site
        by changing input shapes and .json file
        '''
        # NN details for the actor critic algorithm
        self.alpha = 0.05
        self.beta = 0.05
        self.input_dims = [num_turb + 1] # adding 1 to append agc signal
        self.tau = 0.01
        self.batch_size = 40
        self.layer1_size = 2
        self.layer2_size = 2
        self.num_turb = num_turb
        self.num_actions = self.num_turb*2  # yaw and axial for all 9 turbines 
        
        self.fi = fi
        
        # Yaw limits
        self.yaw_low  = 0
        self.yaw_high = 25
        
        # Axial limits
        self.axial_low = 0.1
        self.axial_high = 0.6
        
        # Training parameters
        self.num_episodes = 2
        self.unc = np.linspace(0.9,1.1,self.num_episodes) # uncertainty 
                                                         # if required
        
        # Simulation params
        self.time_delay = 4 # AGC signal updated every 4s
    
    def data(self, df_wind=None, df_AGC_signal=None, use_case=None, 
        train_fraction=0.8):
        '''
        MS: use_case added to cut the data up for training/testing. If 
        use_case=None (default), all data from ts.GetData() will be 
        passed out. Otherwise, only a portion will be past out depending 
        on wether use_case='train' or use_case='test'.
        '''
        time_pow_4s, wind_4s, power_4s =  ts.GetData(
            df_wind=df_wind, df_AGC_signal=df_AGC_signal
        )

        train_length = round(train_fraction*len(time_pow_4s))

        if use_case == None:
            pass
        elif use_case == 'train':
            time_pow_4s = time_pow_4s[:train_length]
            wind_4s = wind_4s[:train_length]
            power_4s = power_4s[:train_length]
        elif use_case == 'test':
            time_pow_4s = time_pow_4s[train_length:]
            wind_4s = wind_4s[train_length:]
            power_4s = power_4s[train_length:]
        else:
            raise ValueError('Invalid use case selected.')

        # end = 7000
        # wind_4s = wind_4s[500:end];
        # power_4s = power_4s[500:end];
        # time_pow_4s = time_pow_4s[500:end]
        
        return wind_4s, power_4s, time_pow_4s


    def train(self, df_wind=None, df_AGC_signal=None, train_fraction=0.8):
        
        wind_4s, power_4s, time_pow_4s = self.data(
            df_wind=df_wind,
            df_AGC_signal=df_AGC_signal,
            use_case='train',
            train_fraction=train_fraction
        )
        
        self.fi.calculate_wake()

        # Agent object:
        agent = Agent(alpha=self.alpha,
                      beta=self.beta,
                      input_dims= self.input_dims,
                      tau= self.tau ,
                      env=env,
                      batch_size=self.batch_size,
                      layer1_size=self.layer1_size,
                      layer2_size=self.layer2_size,
                      n_actions=self.num_actions)

        # Length of the episode:
        self.maxTime = len(power_4s)-1
        # self.maxTime = 3 # FOR TESTING ONLY!
        score_history = []  
        score = 0

        for i in range(self.num_episodes):
            # this obs should also have agc as input
            obs,self.fi = env.reset(self.fi,self.num_turb,wind_4s[0], power_4s[0]) 
            done = False
            #uncFac = self.unc[i];
            uncFac = 1
            t = 0
            step = 0
            a_fil = np.zeros(self.num_actions,)
            while not done:
                
                step += 1
                
                # Choose Action
                act = agent.choose_action(obs)
                
                # Map [-1,1] to actual
                yaw = self.yaw_low + self.yaw_high/2* (act[:self.num_turb]) # yaw
                axial = self.axial_low + self.axial_high/2*(act[self.num_turb:]) #axial
                
                
                # add yaw filter later
                a_fil[:self.num_turb] = yaw
                #a_fil[self.num_turb:] = axial
                
                
                # State Update
                new_state, reward, done, self.fi, power = env.step(a_fil,obs,\
                                                        self.fi,t,power_4s[step]\
                                                        ,uncFac*wind_4s[step],
                                                        self.num_turb)
                
                # Save
                agent.remember(obs, a_fil, reward, new_state, int(done))
                
                # Learn using some optimization algo
                agent.learn()
                
                

                
                # current state = new state
                obs = new_state
                
                
                t += self.time_delay
                
                
                if step >= self.maxTime:
                    done = True;
                    
                    
                score_history.append(reward)
                    
                
        return agent
        
        
    def test(self, agent):
        
        # Get wind and power ref data from 
        wind_4s, power_4s, time_pow_4s = self.data(use_case='test')
        self.maxTime = len(power_4s)-1
        #self.maxTime = 3 # FOR TESTING ONLY!
        
        # initialize floris:
        #fi = wfct.floris_interface.FlorisInterface("datasets/example_input.json")
        self.fi.calculate_wake()
        
        obs,self.fi = env.reset(self.fi,self.num_turb,wind_4s[0], power_4s[0]) 
        done = False
        t = 0
        step = 0
        a_fil = np.zeros(self.num_actions,)
        pwr_list = []
        agc_list = []
        score_history = []
        while not done:
            
            step += 1
            
            # Choose Action
            act = agent.choose_action(obs)
            
            # Map [-1,1] to actual
            yaw = self.yaw_low + self.yaw_high/2* (act[:self.num_turb]) # yaw
            axial = self.axial_low + self.axial_high/2*(act[self.num_turb:]) #axial
            
            
            # add yaw filter later
            a_fil[:self.num_turb] = yaw
            #a_fil[self.num_turb:] = axial
            
            
            # State Update
            new_state, reward, done, self.fi, power = env.step(a_fil,obs,\
                                                    self.fi,t,power_4s[step]\
                                                    ,wind_4s[step],
                                                    self.num_turb)
            
            # Save
            agent.remember(obs, a_fil, reward, new_state, int(done))


            
            # current state = new state
            obs = new_state
            
            
            
            pwr_list.append(power)
            agc_list.append(power_4s[step]*1e6)
            
            
            t += self.time_delay
            
            
            if step >= self.maxTime:
                done = True
                
                
            score_history.append(reward)
            
      
            
            
        plt.figure()
        plt.plot(pwr_list)
        plt.plot(agc_list)
                
            
        return agent
    
    
    
    
    def getControlInput(self, agc, speeds, agent ):
        
        obs = speeds + [agc]
        act = agent.choose_action(obs)
        
        yaw = act[:len(speeds)]
        axial = act[len(speeds):]
        
        return yaw, axial
        
    
    
    
    
    




    