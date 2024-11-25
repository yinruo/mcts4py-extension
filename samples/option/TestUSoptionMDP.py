from enum import Enum
from mcts4py.MDP import *
import numpy as np
from mcts4py.Nodes import *
import math

class TestUSoptionAction(Enum):
    EXERCISE = 0
    HOLD = 1



class TestUSoptionState:
    def __init__(self, time_step: int, asset_price: float, is_terminal: bool, imme: float, expect : float):
        self.time_step = time_step
        self.asset_price = asset_price
        self.is_terminal = is_terminal
        self.imme = imme
        self.expect = expect


class TestUSoptionMDP(MDP[TestUSoptionAction, TestUSoptionState]):
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

    def initial_state(self) -> TestUSoptionState:
        return TestUSoptionState(time_step=0, asset_price=self.S0, is_terminal=False, imme = 0, expect = 0)


    def reward(self, state: TestUSoptionState, action:  TestUSoptionAction, new_state: TestUSoptionState) -> float:
        if action == TestUSoptionAction.EXERCISE:
            return max(self.K - state.asset_price, 0)  
        return 0  
    
    def is_terminal(self, state: TestUSoptionState) -> bool:
        return state.is_terminal or state.time_step == self.T

    def transition(self, state: TestUSoptionState, action: TestUSoptionAction) -> TestUSoptionState:
        if state.time_step == self.T:
            return TestUSoptionState(state.time_step, state.asset_price, True, 0, 0)
        # binimial equation. rasise percentage p, drop percentage 1-p
        #new_price = self.simulate_gbm(self.S0,self.dt, self.r, self.sigma)
        #new_price = state.asset_price * (self.u if np.random.rand() < self.p else self.d)

        if np.random.rand() < self.p:
            new_price = state.asset_price * self.u  
        else:
            new_price = state.asset_price * self.d 

        new_time_step = round(state.time_step + self.dt, 3) 
        if state.time_step + self.dt == self.T or action == TestUSoptionAction.EXERCISE:
            return TestUSoptionState(new_time_step, new_price, True, 0, 0)
        else:
            return TestUSoptionState(new_time_step, new_price, False, 0, 0)
    def actions(self, state: TestUSoptionState) -> list[TestUSoptionAction]:
        return list(TestUSoptionAction)

    def visualize_state(self, state: TestUSoptionState) -> None:
        print(f"Time Step: {state.time_step}, Asset Price: {state.asset_price:.2f}, Terminal: {state.is_terminal}")

    def print_tree(self, node: ActionNode[TState, TAction], level=0, is_last=True, prefix=""):
        """Recursive tree printing, showing both asset price and option price at each node"""
        indent = prefix + ("|-- " if is_last else "|-- ")
        
        # Print both asset price and option price at each node
        print(f"{indent}asset price: {node.state.asset_price:.4f}, action:{node.inducing_action}, time: {node.state.time_step} ")

        prefix += "    " if is_last else "|   "
        
        # Recursively print children nodes
        num_children = len(node.children)
        for idx, child in enumerate(node.children):
            is_last_child = (idx == num_children - 1)
            self.print_tree(child, level + 1, is_last_child, prefix)
