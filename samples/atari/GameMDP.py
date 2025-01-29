from enum import Enum
from mcts4py.MDP import *
import gymnasium as gym
from samples.atari.config import ACTIONS, game_name

class GameState:
    def __init__(self, state, is_terminal: bool):
        self.current_state = state
        self.is_terminal = is_terminal

GameAction = Enum("GameAction", ACTIONS[f"ALE/{game_name}"])
    
class GameMDP(MDP[GameAction, GameState]):
    def __init__(self):
        self.env_name = f"ALE/{game_name}"
        self.env = gym.make(self.env_name)

    def initial_state(self) -> GameState:
        self.env.reset()
        initial_state = self.env.unwrapped.clone_state(include_rng=True)
        return GameState(initial_state, is_terminal=False)

    def reward(self, state: GameState, action:  GameAction, new_state: GameState) -> float:
        self.env.unwrapped.restore_state(state.current_state)
        observation, reward, terminated, truncated, _ = self.env.step(action.value)
        return reward
    
    def is_terminal(self, state: GameState) -> bool:
        return state.is_terminal

    def transition(self, state: GameState, action: GameAction) -> GameState:
        self.env.unwrapped.restore_state(state.current_state)
        observation, reward, terminated, truncated, _ = self.env.step(action.value)
        done = terminated or truncated
        new_state = self.env.unwrapped.clone_state(include_rng=True)
        return GameState(new_state, done)

    def actions(self) -> list[GameAction]:
        return list(GameAction)

    def visualize_state(self, state: GameState) -> None:
        state.env.render()
