from enum import Enum
from mcts4py.MDP import *
import numpy as np
import gymnasium as gym
from copy import deepcopy
from typing import Tuple

class cartpoleAction(Enum):
    LEFT = 0
    RIGHT = 1


class cartpoleState:
    def __init__(self, env: gym.Env, is_terminal: bool):
        self.env = env
        self.is_terminal = is_terminal


class cartpoleMDP(MDP[cartpoleState, cartpoleAction]):
    def __init__(self,
                 env_name: str = 'CartPole-v1'
                 ):
        self.env = gym.make(env_name)
        self.env.reset()

    def initial_state(self) -> cartpoleState:
        observation, info = self.env.reset() 
        initial_env = deepcopy(self.env) 
        return cartpoleState(initial_env, is_terminal=False)

    def reward(self, state: cartpoleState, action:  cartpoleAction, new_state: cartpoleState) -> float:
        env = state.env
        observation, reward, terminated, truncated, _ = env.step(action.value)
        return reward
    
    def is_terminal(self, state: cartpoleState) -> bool:
        return state.is_terminal

    def transition(self, state: cartpoleState, action: cartpoleAction) -> cartpoleState:
        new_env = deepcopy(state.env)
        observation, reward, terminated, truncated, _ = new_env.step(action.value)
        done = terminated or truncated
        return cartpoleState(new_env, done)

    def actions(self, state: cartpoleState) -> list[cartpoleAction]:
        return [cartpoleAction.LEFT, cartpoleAction.RIGHT]

    def visualize_state(self, state: cartpoleState) -> None:
        state.env.render()
