import numpy as np
import math
from mcts4py.MDP import *
from mcts4py.Solver import *
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy

class SolverMENTS_copy(MCTSSolver[TAction, MENTSNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):
    def __init__(self, 
                 mdp: MDP[TState, TAction], 
                 simulation_depth_limit, 
                 exploration_constant, 
                 discount_factor, 
                 temperature, 
                 epsilon,
                 env_name,
                 verbose: bool = False):
        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.epsilon = epsilon
        self.env_name = env_name
        #self.__root_node = MENTSNode[TState, TAction](None, None)
        #self.simulate_action(self.__root_node)
        #self.env = gym.make(self.env_name)
        #self.env.reset()

        super().__init__(exploration_constant, verbose)

    def reset_node(self, node: MENTSNode[TState, TAction]):
        # Reset the node's statistics
        node.n = defaultdict(float)  
        for action in node.valid_actions:
            node.n[action.value] = 0.0
        node.max_reward = float('-inf')
        node.depth = 0.0
        node.Q_sft = defaultdict(float)  
        node.reward = defaultdict(float)
        for action in node.valid_actions:
            node.Q_sft[action.value] = 0.0

    def run_game(self, episodes: int):
        total_reward = 0
        for e in range(episodes):
            reward_episode = 0
            done = False
            root_node = MENTSNode[TState, TAction](None, None)
            self.simulate_action(root_node)
            self.reset_node(root_node)
            root_node.state = self.mdp.initial_state() 
            game = gym.make(self.env_name)
            game.reset()
            root_node.state.env = deepcopy(game.unwrapped)
            print('episode #' + str(e+1))
            current_node = root_node
            while not done: 
                next_node, action = self.run_iteration(current_node, 100)
                print(action.value)
                observation, reward, terminated, truncated, _ = game.step(action.value)
                reward_episode += reward
                print("reward for episode " + str(e+1), reward_episode)
                done = terminated or truncated
                current_node = next_node

                if done:
                    print('reward ' + str(reward_episode))
                    total_reward += reward_episode 
                    game.close()
                    break
        average_reward = total_reward / episodes if episodes > 0 else 0
        return average_reward

    def run_iteration(self, node, iterations):
        for i in range(iterations):
            if self.verbose:
                print("Select start")
            selected_node = self.select(node)
            if self.verbose:
                print("Expand start")
            expanded, action = self.expand(selected_node)
            if action == None:
                continue
            if self.verbose:
                print("Simulate start")
            R = self.simulate(expanded)
            if self.verbose:
                print("Backpropagate start")
            self.backpropagate(selected_node, action, R)

        if not node.children:
            print("No children added to root node after iterations")
        children = node.children
        best_child = max(children, key=lambda c: c.Q_sft[c.inducing_action.value])
        return best_child, best_child.inducing_action
    
    def softmax_value(self, Q_values):
        temp = self.temperature
        Q_values_array = np.array(list(Q_values.values()))
        max_Q = np.max(Q_values_array) 
        exp_values = np.exp((Q_values_array - max_Q) / temp) 
        sum_exp_values = np.sum(exp_values)  
        softmax_val = temp * np.log(sum_exp_values)+ max_Q  
        return softmax_val
    
    def soft_indmax(self, Q_values):
        softmax = self.softmax_value(Q_values)
        Q_values_array = np.array(list(Q_values.values()))
        # Calculate fτ(r) using the formula exp((r - Fτ(r)) / τ)
        soft_indmax_value = np.exp((Q_values_array - softmax) / self.temperature)
    
        return soft_indmax_value

    def e2w(self, node: MENTSNode[TState, TAction]):
        total_visits = 0
        for action in node.valid_actions:
            total_visits += node.n[action.value]
        if total_visits == 0:
            lambda_s = 1.0 
        else:
            lambda_s = (self.epsilon * len(node.valid_actions)) / np.log(total_visits + 1)
        if self.verbose:
                print("node.Q-sft:", node.Q_sft)
        soft_indmax_probs = self.soft_indmax(node.Q_sft)
        action_probabilities = (1 - lambda_s) * soft_indmax_probs + lambda_s * (1 / len(node.valid_actions))
        if self.verbose:        
            print("action_probabilities", action_probabilities)
        action = np.random.choice(node.valid_actions, p=action_probabilities)
        return action
    
    def root(self) -> MENTSNode[TState, TAction]:
        return self.__root_node
    
    def simulate_action(self, node: MENTSNode[TState, TAction]):
        if node.parent == None:
            initial_state = self.mdp.initial_state()
            node.state = initial_state
            node.valid_actions = self.mdp.actions(node.state)
            return

        if node.inducing_action == None:
            raise RuntimeError("Action was null for non-null parent")
        new_state = self.mdp.transition(node.parent.state, node.inducing_action)
        node.state = new_state
        node.valid_actions = self.mdp.actions(node.state)
    
    def select(self, node: MENTSNode[TState, TAction]):
        if len(node.children) == 0:
            return node

        current_node = node
        self.simulate_action(node)

        while True:
            # If the node is terminal, return it
            if self.mdp.is_terminal(current_node.state):
                return current_node

            current_children = current_node.children
            explored_actions = set([c.inducing_action for c in current_children])

            # This state has not been fully explored, return it
            if len(set(current_node.valid_actions) - explored_actions) > 0:
                return current_node
            
            action = self.e2w(current_node)
            child_node = next((child for child in current_children if child.inducing_action == action), None)
            if self.verbose:
                print("child_node:", child_node)
                print("node children", node.children)
            if child_node:
                # Continue looping with this child node
                current_node = child_node

    def expand(self, node: MENTSNode[TState, TAction]):
        # If the node is terminal, return it
        if node.state.is_terminal:
            return node, None

        current_children = node.children
        explored_actions = set([c.inducing_action for c in current_children])
        valid_action: set[TAction] = set(node.valid_actions)
        unexplored_actions = valid_action - explored_actions
        if self.verbose:
            print("unexplored_actions: ",unexplored_actions)

        # Expand an unexplored action
        action_taken = random.sample(list(unexplored_actions), 1)[0]
        if self.verbose:
            print("selected action: ",action_taken)

        new_node = MENTSNode(node, action_taken)
        node.add_child(new_node)
        self.simulate_action(new_node)
        self.reset_node(new_node)
        new_game = deepcopy(node.state.env)
        observation, r_reward, terminated, truncated, _ = new_game.step(action_taken.value)
        node.reward[action_taken.value] = r_reward
        return new_node, action_taken


    def simulate(self, node):
        new_game = deepcopy(node.state.env)
        parent_node = node.parent
        done = False
        R = 0
        while not done:
            random_action = random.choice(node.valid_actions)
            observation, r_reward, terminated, truncated, _ = new_game.step(random_action.value)
            done = terminated or truncated
            R += r_reward
            if done:
                if self.verbose:
                    print("future reward:",R)
                return R

    def backpropagate(self, node, action, R):
        node.n[action.value] += 1
        node.Q_sft[action.value] = node.reward[action.value] + R
        inducing_action = node.inducing_action
        softmax_value = self.softmax_value(node.Q_sft)
        node = node.parent

        while node: 
            # Softmax backup
            node.n[inducing_action.value] += 1
            node.Q_sft[inducing_action.value] = node.reward[inducing_action.value] + softmax_value
            if self.verbose:
                print("softmax value:", softmax_value)
            if self.verbose:
                print("Q_sft:", node.Q_sft)
            softmax_value = self.softmax_value(node.Q_sft)
            node = node.parent








        