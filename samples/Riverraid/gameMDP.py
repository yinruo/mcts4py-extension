from enum import Enum
from mcts4py.MDP import *
import numpy as np
import gymnasium as gym
from copy import deepcopy
from typing import Tuple

class RiverraidAction(Enum):
    LEFT = 0
    RIGHT = 1


class RiverraidState:
    def __init__(self, state, is_terminal: bool):
        self.current_state = state
        self.is_terminal = is_terminal


class gameMDP(MDP[RiverraidAction, RiverraidState]):
    def __init__(self,
                 env_name: str = 'CartPole-v1'
                 ):
        self.env = gym.make(env_name)

    def initial_state(self) -> RiverraidState:
        self.env.reset()
        initial_state = self.env.unwrapped.clone_state(include_rng=True)
        return RiverraidState(initial_state, is_terminal=False)

    def reward(self, state: RiverraidState, action:  RiverraidAction, new_state: RiverraidState) -> float:
        self.env.unwrapped.restore_state(state.current_state)
        observation, reward, terminated, truncated, _ = self.env.step(action.value)
        return reward
    
    def is_terminal(self, state: RiverraidState) -> bool:
        return state.is_terminal

    def transition(self, state: RiverraidState, action: RiverraidAction) -> RiverraidState:
        self.env.unwrapped.restore_state(state.current_state)
        observation, reward, terminated, truncated, _ = self.env.step(action.value)
        done = terminated or truncated
        new_state = self.env.unwrapped.clone_state(include_rng=True)
        return RiverraidState(new_state, done)

    def actions(self) -> list[RiverraidAction]:
        return list(RiverraidAction)

    def visualize_state(self, state: RiverraidState) -> None:
        state.env.render()
