from enum import Enum
from mcts4py.MDP import *
import numpy as np
import gymnasium as gym
from copy import deepcopy
from typing import Tuple

class lunarlanderAction(Enum):
    NO = 0
    LEFT = 1
    MAIN = 2
    RIGHT = 3

class lunarlanderState:
    def __init__(self, env: gym.Env, is_terminal: bool):
        self.env = env
        self.is_terminal = is_terminal


class lunarlanderMDP(MDP[lunarlanderState, lunarlanderAction]):
    def __init__(self,
                 env_name: str = 'LunarLander-v2'
                 ):
        self.env = gym.make(env_name)
        self.env.reset()

    def initial_state(self) -> lunarlanderState:
        observation, info = self.env.reset() 
        initial_env = deepcopy(self.env.unwrapped) 
        return lunarlanderState(initial_env, is_terminal=False)

    def reward(self, state: lunarlanderState, action:  lunarlanderAction, new_state: lunarlanderState) -> float:
        env = state.env
        observation, reward, terminated, truncated, _ = env.step(action.value)
        return reward
    
    def is_terminal(self, state: lunarlanderState) -> bool:
        return state.is_terminal

    def transition(self, state: lunarlanderState, action: lunarlanderAction) -> lunarlanderState:
        new_env = deepcopy(state.env.unwrapped)
        observation, reward, terminated, truncated, _ = new_env.step(action.value)
        done = terminated or truncated
        return lunarlanderState(new_env, done)

    def actions(self, state: lunarlanderState) -> list[lunarlanderAction]:
        return [lunarlanderAction.LEFT, lunarlanderAction.RIGHT]

    def visualize_state(self, state: lunarlanderState) -> None:
        state.env.render()
