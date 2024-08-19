import gymnasium as gym
from copy import deepcopy
from mcts4py.GenericSolarisSolver import *
from mcts4py.StatefulSolver import *
from mcts4py.GenericSolver import GenericSolver
from samples.solaris.solarisMDP import *


mdp = solarisMDP()

solver = GenericSolarisSolver(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 0.5,
    verbose = False)


solver.run_game(10) 


""" import gymnasium as gym
rewards = []
moving_average = []
for e in range(10):
    reward_episode = 0
    done = False
    game = gym.make('ALE/Solaris-v5', render_mode="human")
    game.reset()
    print('episode #' + str(e+1))

    while not done:
        action = game.action_space.sample()
        observation, reward, terminated, truncated, _ = game.step(action)
        reward_episode += reward
        done = terminated or truncated

        if done:
            print('reward ' + str(reward_episode))
            game.close()
            break """


