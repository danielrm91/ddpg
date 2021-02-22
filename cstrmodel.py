## CSTR MODEL
## 23/01/21
##
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import numpy as np
import random
from scipy.integrate import odeint


class cstr_model(gym.Env):
    
    def __init__(self):
        
        self.max_Ca = 0.3
        self.max_Qchange = 100
        self.rho = 780 #kg/m3    
        self.Cp = 3.25 #kJ/kgK
        self.EaA = 41570 #Kj/kmol
        self.EaB = 45727 #Kj/kmol
        self.F = 0.0025 #m3/min
        self.DH = 4157 #Kj/kmol
        self.k0A = 50000 # 1/min
        self.k0B = 100000 #1/min
        self.R = 8.314 #Kj/kmolK
        self.V = 0.2 #m3
        self.Ca_ss = 0.3 
        self.Cb_ss = 0.0
        self.T_ss = 300
        self.Tf = 300
        
        #high = np.array([1.,1.,self.max_Ca],dtype=np.float32)
        high = np.array([0.3, np.inf, np.inf])
        low = np.array([0.0, 0.0, 300])
        
        self.action_space = spaces.Box(
            low = 0.0,
            high= self.max_Qchange, shape=(1,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low= low,
            high=high,
            dtype=np.float32
        )
        
        self.seed()
        
       
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return[seed]
    
    def time(self, j, max_steps):
        self.t = j
        self.max_steps = max_steps
        
    def befstate(self, s):
        self.s = s

#step
    def step(self, a):               
        
        if self.t == 0 :
            self.Q = 4999
        else:
            self.Q = self.Q + a
            
        self.Q = np.clip(self.Q, 0 , 5000)#[0]      
        
        def cstr(x,t,Q,Tf,Caf):  
            
            Ca, Cb, Temp = self.state             
            
            self.state[0] = x[0]
            self.state[1] = x[1]
            self.state[2] = x[2]
            
            self.ka = self.k0A * np.exp(-self.EaA / (self.R * Temp))
            self.kb = self.k0B * np.exp(-self.EaB / (self.R * Temp))
            self.rA = self.ka * Ca * self.V
            self.rB = self.kb * Cb * self.V
            
            # Calculate concentration derivative
            dCadt = (self.F * (self.Ca_ss - Ca) - self.rA + self.rB) / self.V
            dCbdt = ((self.F * -Cb) + self.rA - self.rB) / self.V
            # Calculate temperature derivative
            dTempdt = ((self.F * self.rho * self.Cp) * (self.T_ss - Temp) + self.Q - self.DH * (self.rA - self.rB)) / (self.rho * self.Cp * self.V)
            
            # Return xdot:
            xdot = np.zeros(3)
            xdot[0] = dCadt
            xdot[1] = dCbdt
            xdot[2] = dTempdt
            return xdot
                
        self.x0 = np.empty(3)
        self.x0[0] = self.s[0]
        self.x0[1] = self.s[1]
        self.x0[2] = self.s[2]
               
        ts = [self.t,self.t+1]
        
        y = odeint(cstr,self.x0,ts,args=(self.Q,self.Tf,self.Ca_ss))
        
        newCa = y[-1][0]
        newCb = y[-1][1]
        newTemp = y[-1][2]
        self.x0[0] = newCa
        self.x0[1] = newCb
        self.x0[2] = newTemp        
                             
        costs = newCb
        
        # store the new observation for the state
        self.state = np.array([newCa, newCb, newTemp])
        
        #print(newCa, newCb, newTemp, self.Q, a)
        
        #aqui estamos guardando todos los datos generados en cada step de una corrida del environment
        g = open("data.txt", "a+")
        g.write(str(self.t) + " " + str(costs) +  " " +str(self.Q) + '\n')
                
        # check if the episode is over and store as "done"
        return self._get_obs(), costs, False, {}
        
    def _get_obs(self):
        Ca, Cb, Temp = self.state
        return np.array([Ca, Cb, Temp])
    
# reset method

    def reset(self):
        high = np.array([0.3, 0.0, 300])
        low = np.array([0.0, 0.0, 300])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs()
    
# render
   
    def render (self):
        data = np.vstack(([j],self.Q,Temp)) # vertical stack
        data = data.Temp            # transpose data
        np.savetxt('data_doublet.txt',data,delimiter=',',\
                   header='Time,Q,T',comments='')
        

        
        
        
    
    
    
    
   