from enum import Enum
from mcts4py.MDP import *
import numpy as np
from mcts4py.Nodes import *
import math

class USoptionAction(Enum):
    UP = 0 
    DOWN = 1


class USoptionState:
    def __init__(self, time_step: int, asset_price: float, is_terminal: bool, imme: float, expect : float):
        self.time_step = time_step
        self.asset_price = asset_price
        self.is_terminal = is_terminal
        self.imme = imme
        self.expect = expect


class USoptionMDP(MDP[USoptionAction, USoptionState]):
    def __init__(self, 
                 option_type,
                 S0: float,  # initial price
                 K: float,  # strike price
                 r: float,  
                 T: float,   
                 dt: float,
                 sigma: float
                 ): 
        self.option_type = option_type   
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.dt = dt
        self.sigma = sigma 
        self.u = np.exp(sigma * np.sqrt(self.dt)) 
        self.d = 1 / self.u  
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)  

    def simulate_gbm(self, S, dt, r, sigma):
        Z = np.random.normal()
        return S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * Z)

    def initial_state(self) -> USoptionState:
        return USoptionState(time_step=0, asset_price=self.S0, is_terminal=False, imme = 0, expect = 0)


    def reward(self, state: USoptionState, action:  USoptionAction, new_state: USoptionState) -> float:
        if action == USoptionAction.EXERCISE:
            return max(self.K - state.asset_price, 0)  
        return 0  
    
    def is_terminal(self, state: USoptionState) -> bool:
        return state.is_terminal or state.time_step == self.T

    def transition(self, state: USoptionState, action: USoptionAction) -> USoptionState:
        if state.time_step == self.T:
            return USoptionState(state.time_step, state.asset_price, True, 0, 0)
        # binimial equation. rasise percentage p, drop percentage 1-p
        #new_price = self.simulate_gbm(self.S0,self.dt, self.r, self.sigma)
        #new_price = state.asset_price * (self.u if np.random.rand() < self.p else self.d)
        if action == USoptionAction.UP:
            new_price = state.asset_price * self.u
        elif action == USoptionAction.DOWN:
            new_price = state.asset_price * self.d 
        if state.time_step == 4.5:
            return USoptionState(state.time_step + self.dt, new_price, True, 0, 0)
        else:
            return USoptionState(state.time_step + self.dt, new_price, False, 0, 0)
    def actions(self, state: USoptionState) -> list[USoptionAction]:
        return list(USoptionAction)

    def visualize_state(self, state: USoptionState) -> None:
        print(f"Time Step: {state.time_step}, Asset Price: {state.asset_price:.2f}, Terminal: {state.is_terminal}")

