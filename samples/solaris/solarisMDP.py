from enum import Enum
from mcts4py.MDP import *
import numpy as np
import gymnasium as gym
from copy import deepcopy
from typing import Tuple

class solarisAction(Enum):
    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    UPRIGHT = 6
    UPLEFT = 7
    DOWNRIGHT = 8
    DOWNLEFT = 9
    UPFIRE = 10
    RIGHTFIRE = 11
    LEFTFIRE = 12
    DOWNFIRE = 13
    UPRIGHTFIRE = 14
    DOWNLEFTFIRE = 15
    UPLEFTFIRE = 16
    DOWNRIGHTFIRE = 17


class solarisState:
    def __init__(self, state, is_terminal: bool):
        self.current_state = state
        self.is_terminal = is_terminal


class solarisMDP(MDP[solarisState, solarisAction]):
    def __init__(self,
                 env_name: str = 'ALE/Riverraid-v5'
                 ):
        self.env = gym.make(env_name)
        self.env.reset()

    def initial_state(self) -> solarisState:
        observation, info = self.env.reset() 
        initial_state = self.env.unwrapped.clone_state(include_rng=True)
        return solarisState(initial_state, is_terminal=False)

    def reward(self, state: solarisState, action:  solarisAction, new_state: solarisState) -> float:
        self.env.unwrapped.restore_state(state.current_state)
        observation, reward, terminated, truncated, _ = self.env.step(action.value)
        return reward
    
    def is_terminal(self, state: solarisState) -> bool:
        return state.is_terminal

    def transition(self, state: solarisState, action: solarisAction) -> solarisState:
        self.env.unwrapped.restore_state(state.current_state)
        observation, reward, terminated, truncated, _ = self.env.step(action.value)
        done = terminated or truncated
        new_state = self.env.unwrapped.clone_state(include_rng=True)
        return solarisState(new_state, done)

    def actions(self, state: solarisState) -> list[solarisAction]:
        return list(solarisAction)

    def visualize_state(self, state: solarisState) -> None:
        state.env.render()
