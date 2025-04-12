from enum import Enum
from mcts4py.MDP import *
import numpy as np
from mcts4py.Nodes import *
import math
from typing import List
from sklearn.mixture import GaussianMixture

class MultiOptionAction:
    def __init__(self, decisions: List[int]): 
        self.decisions = decisions

class MultiOptionState:
    def __init__(self, time_step: float, asset_prices: List[float], exercised_flags: List[bool], is_terminal: bool):
        self.time_step = time_step
        self.asset_prices = asset_prices
        self.exercised_flags = exercised_flags
        self.is_terminal = is_terminal


class USMultiOptionMDP(MDP[MultiOptionAction, MultiOptionState]):
    def __init__(self, 
                 option_type_list:List[str],
                 S0_list: List[float], 
                 K_list: List[float], 
                 r: float,  
                 T: float,   
                 dt: float,
                 sigma_list: List[float],
                 q_list: List[float],
                 max_exercise_per_step: int,
                 ): 
        self.n = len(S0_list)
        self.option_type_list = option_type_list
        self.S0_list = S0_list
        self.K_list = K_list
        self.r = r
        self.T = T
        self.dt = dt
        self.sigma_list = sigma_list
        self.q_list = q_list
        self.max_exercise_per_step = max_exercise_per_step
        self.u_list = [np.exp(s * np.sqrt(dt)) for s in sigma_list]
        self.d_list = [1.0 / u for u in self.u_list]
        self.p_list = [(np.exp((r - q) * dt) - d) / (u - d) for u, d, q in zip(self.u_list, self.d_list, q_list)]
        self.gmm_model = GaussianMixture(n_components=1)
        fake_data = np.random.normal(0, 0.2, size=(1000, self.n))
        self.gmm_model.fit(fake_data)
        self.corr_matrix = np.identity(self.n)
        sigma_array = np.array(self.sigma_list)
        self.cov_matrix = np.outer(sigma_array, sigma_array) * self.corr_matrix * self.dt

    def get_intrinsic_value(self, S, K, option_type):
        if option_type == "put":
            return np.maximum(K - S, 0)  
        elif option_type == "call":
            return np.maximum(S - K, 0) 
        else:
            print("option type unknown")

    def simulate_gmm(self, current_prices: List[float]) -> List[float]:
        log_returns = self.gmm_model.sample()[0][0]
        new_prices = [S * np.exp(r) for S, r in zip(current_prices, log_returns)]
        return new_prices
    
    def simulate_mvbm(self, current_prices: List[float]) -> List[float]:
        mu = np.array([(self.r - q - 0.5 * sigma ** 2) * self.dt for q, sigma in zip(self.q_list, self.sigma_list)])
        cov = self.cov_matrix 
        log_returns = np.random.multivariate_normal(mean=mu, cov=cov)
        new_prices = [S * np.exp(r) for S, r in zip(current_prices, log_returns)]
        return new_prices

    def initial_state(self) -> MultiOptionState:
        return MultiOptionState(
            time_step=0.0,
            asset_prices=self.S0_list[:],
            exercised_flags=[False] * self.n,
            is_terminal=False
        )


    def reward(self, state: MultiOptionState, action: MultiOptionAction, new_state: MultiOptionState) -> float:
        reward = 0.0
        for i, do_exercise in enumerate(action.decisions):
            if do_exercise and not state.exercised_flags[i]:
                reward += self.get_intrinsic_value(
                    state.asset_prices[i], 
                    self.K_list[i], 
                    self.option_type_list[i]
                )
        return reward
    
    def is_terminal(self, state: MultiOptionState) -> bool:
        return state.is_terminal or all(state.exercised_flags) or state.time_step >= self.T

    def transition(self, state: MultiOptionState, action: MultiOptionAction) -> MultiOptionState:
        if self.is_terminal(state):
            return MultiOptionState(state.time_step, state.asset_prices, state.exercised_flags, True)
        new_prices = self.simulate_mvbm(state.asset_prices)
        new_flags = [
            True if old or act else False 
            for old, act in zip(state.exercised_flags, action.decisions)
        ]
        new_time = round(state.time_step + self.dt, 5)
        done = (new_time >= self.T) or all(new_flags)
        return MultiOptionState(new_time, new_prices, new_flags, done)
        
    def actions(self, state: MultiOptionState) -> List[MultiOptionAction]:
        possible_actions = []
        for bits in range(2**self.n):
            decisions = [(bits >> i) & 1 for i in range(self.n)]
            if any(dec and state.exercised_flags[i] for i, dec in enumerate(decisions)):
                continue
            if sum(decisions) > self.max_exercise_per_step:
                continue
            possible_actions.append(MultiOptionAction(decisions))
        return possible_actions

