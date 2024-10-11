from enum import Enum
from mcts4py.MDP import *
import numpy as np
import gymnasium as gym
from copy import deepcopy
from typing import Tuple

class EUoptionAction(Enum):
    HOLD = 0 
    EXERCISE = 1


class EUoptionState:
    def __init__(self, time_step: int, asset_price: float, is_terminal: bool):
        self.time_step = time_step
        self.asset_price = asset_price
        self.is_terminal = is_terminal


class EUoptionMDP(MDP[EUoptionAction, EUoptionState]):
    def __init__(self, 
                 S0: float,  # initial price
                 K: float,  # strike price
                 r: float,  
                 T: float,   
                 sigma: float, 
                 n: int):    
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.n = n
        self.delta_t = T / n 
        self.u = np.exp(sigma * np.sqrt(self.delta_t))  # asset raise percentage
        self.d = 1 / self.u  
        self.p = (np.exp(r * self.delta_t) - self.d) / (self.u - self.d)  

    def initial_state(self) -> EUoptionState:
        return EUoptionState(time_step=0, asset_price=self.S0, is_terminal=False)


    def reward(self, state: EUoptionState, action:  EUoptionAction, new_state: EUoptionState) -> float:
        #if action == EUoptionAction.EXERCISE and state.time_step == self.n:
            #return max(self.K - state.asset_price, 0)  
        if state.time_step == self.n :
            if action == EUoptionAction.EXERCISE:
                return max(self.K - state.asset_price, 0)
            else:
                return 0
        return 0  
    
    def is_terminal(self, state: EUoptionState) -> bool:
        return state.time_step == self.n + 1

    def transition(self, state: EUoptionState, action: EUoptionAction) -> EUoptionState:
        if state.time_step == self.n:
            if action == EUoptionAction.EXERCISE:
                return EUoptionState(state.time_step+1, state.asset_price, True)
            return EUoptionState(state.time_step+1, state.asset_price, True)
        # binimial equation. rasise percentage p, drop percentage 1-p
        new_price =state.asset_price * (self.u if np.random.rand() < self.p else self.d)
        return EUoptionState(state.time_step + 1, new_price, False)

    def actions(self, state: EUoptionState) -> list[EUoptionAction]:
        if state.time_step > self.n:
            return None 
        
        if state.time_step == self.n:
            return list(EUoptionAction)  
        
        return [EUoptionAction.HOLD]

    def visualize_state(self, state: EUoptionState) -> None:
        print(f"Time Step: {state.time_step}, Asset Price: {state.asset_price:.2f}, Terminal: {state.is_terminal}")


if __name__ == "__main__":
    mdp = EUoptionMDP(S0=100, K=100, r=0.05, T=1, sigma=0.2, n=20)
    state = mdp.initial_state()
    
    mdp.visualize_state(state)
    
    while not state.is_terminal:
        action = EUoptionAction.HOLD
        new_state = mdp.transition(state, action)
        reward = mdp.reward(state, action, new_state)
        mdp.visualize_state(new_state)
        state = new_state