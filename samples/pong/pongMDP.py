from enum import Enum
from mcts4py.MDP import *
import numpy as np
import gymnasium as gym
from copy import deepcopy
from typing import Tuple

class pongAction(Enum):
    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3
    RIGHTFIRE = 4
    LEFTFIRE = 5



class pongState:
    def __init__(self, state, is_terminal: bool):
        self.current_state = state
        self.is_terminal = is_terminal


class pongMDP(MDP[pongAction, pongState]):
    def __init__(self,
                 env_name: str = 'ALE/Riverraid-v5'
                 ):
        self.env = gym.make(env_name)

    def initial_state(self) -> pongState:
        self.env.reset()
        initial_state = self.env.unwrapped.clone_state(include_rng=True)
        return pongState(initial_state, is_terminal=False)

    def reward(self, state: pongState, action:  pongAction, new_state: pongState) -> float:
        self.env.unwrapped.restore_state(state.current_state)
        observation, reward, terminated, truncated, _ = self.env.step(action.value)
        return reward
    
    def is_terminal(self, state: pongState) -> bool:
        return state.is_terminal

    def transition(self, state: pongState, action: pongAction) -> pongState:
        self.env.unwrapped.restore_state(state.current_state)
        observation, reward, terminated, truncated, _ = self.env.step(action.value)
        done = terminated or truncated
        new_state = self.env.unwrapped.clone_state(include_rng=True)
        return pongState(new_state, done)

    def actions(self) -> list[pongAction]:
        return list(pongAction)

    def visualize_state(self, state: pongState) -> None:
        state.env.render()
