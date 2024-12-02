from enum import Enum
from mcts4py.MDP import *
from mcts4py.Nodes import *
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
                 dt: float,
                 ):    
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.dt = dt
        self.u = np.exp(sigma * np.sqrt(self.dt))  # asset raise percentage
        self.d = 1 / self.u  
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)  

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
        new_price =state.asset_price * (self.u if np.random.rand() < self.p else self.d)
        if state.time_step == self.T:
            if action == EUoptionAction.EXERCISE:
                return EUoptionState(state.time_step+1, new_price, True)
            return EUoptionState(state.time_step+1, new_price, True)
        # binimial equation. rasise percentage p, drop percentage 1-p
        return EUoptionState(state.time_step + 1, new_price, False)

    def actions(self, state: EUoptionState) -> list[EUoptionAction]:
        if state.time_step + self.dt == self.T:
            return list(EUoptionAction)  
        
        return [EUoptionAction.HOLD]

    def visualize_state(self, state: EUoptionState) -> None:
        print(f"Time Step: {state.time_step}, Asset Price: {state.asset_price:.2f}, Terminal: {state.is_terminal}")

    def print_tree(self, node: ActionNode[TState, TAction], level=0, is_last=True, prefix=""):
        """Recursive tree printing, showing both asset price and option price at each node"""
        indent = prefix + ("|-- " if is_last else "|-- ")
        
        # Print both asset price and option price at each node
        print(f"{indent}asset price: {node.state.asset_price:.4f}, action:{node.inducing_action} time: {node.state.time_step} ")

        prefix += "    " if is_last else "|   "
        
        # Recursively print children nodes
        num_children = len(node.children)
        for idx, child in enumerate(node.children):
            is_last_child = (idx == num_children - 1)
            self.print_tree(child, level + 1, is_last_child, prefix)

""" if __name__ == "__main__":
    mdp = EUoptionMDP(S0=100, K=100, r=0.05, T=1, sigma=0.2, n=20)
    state = mdp.initial_state()
    
    mdp.visualize_state(state)
    
    while not state.is_terminal:
        action = EUoptionAction.HOLD
        new_state = mdp.transition(state, action)
        reward = mdp.reward(state, action, new_state)
        mdp.visualize_state(new_state)
        state = new_state """

if __name__ == "__main__":
    mdp = EUoptionMDP(S0=1, K=0.95, r=0.05, T=10, sigma=0.2, dt=1)
    root_node = ActionNode[TState, TAction](None, None)
    root_node.state = mdp.initial_state()
    current_node = root_node
    while True:
        actions = mdp.actions(current_node.state)
        for action in actions:
            new_node = ActionNode(current_node, action)
            current_node.add_child(new_node)
            new_state = mdp.transition(current_node.state, action)
            new_node.state = new_state
        if new_node.state.time_step == mdp.T:
            break
        current_node = new_node
    # Visualize the tree
    mdp.print_tree(root_node)