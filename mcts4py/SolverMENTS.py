import numpy as np
import math
from mcts4py.MDP import *
from mcts4py.Solver import *
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict

class SolverMENTS(MCTSSolver[TAction, MENTSNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):
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
        self.__root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(self.__root_node)
        self.env = gym.make(self.env_name)
        self.env.reset()

        super().__init__(exploration_constant, verbose)

    def reset_node(self, node: MENTSNode[TState, TAction]):
        # Reset the node's statistics
        node.n = 0
        node.max_reward = float('-inf')
        node.depth = 0.0
        node.Q_sft = defaultdict(float)  
        node.reward = defaultdict(float)
        node.future_reward = 0.0
        for action in node.valid_actions:
            node.Q_sft[action.value] = 0.0

    def run_game(self, episodes: int):
        actions = []
        rewards = []
        total_reward = 0
        for e in range(episodes):
            reward_episode = 0
            done = False
            root_node = MENTSNode[TState, TAction](None, None)
            self.simulate_action(root_node)
            self.reset_node(root_node)
            root_node.state = self.mdp.initial_state() 
            print(root_node)
            initial_s = self.env.unwrapped.clone_state(include_rng=True)
            print('episode #' + str(e+1))

            while not done: 
                root_node, action = self.run_iteration(root_node, 30)
                print(action.value)
                observation, reward, terminated, truncated, _ = self.env.step(action.value)
                actions.append(action)
                reward_episode += reward
                done = terminated or truncated

                if done:
                    print('reward ' + str(reward_episode))
                    rewards.append(reward_episode)
                    total_reward += reward_episode 
                    self.env.close()
                    break
        average_reward = total_reward / episodes if episodes > 0 else 0
        return rewards, average_reward

    def run_iteration(self, node, iterations):
        current_state = self.env.unwrapped.clone_state(include_rng=True)
        for i in range(iterations):
            self.env.unwrapped.restore_state(current_state)
            if self.verbose:
                print("Select start")
            selected_node, selected_action = self.select(node)
            if self.verbose:
                print("Expand start")
            expanded = self.expand(selected_node, selected_action)
            if self.verbose:
                print("Simulate start")
            self.simulate(expanded)
            if self.verbose:
                print("Backpropagate start")
            self.backpropagate(selected_node, selected_action)

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
        total_visits = node.n
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
            node.valid_actions = self.mdp.actions()
            return

        if node.inducing_action == None:
            raise RuntimeError("Action was null for non-null parent")
        new_state = self.mdp.transition(node.parent.state, node.inducing_action)
        node.state = new_state
        node.valid_actions = self.mdp.actions()
    
    def select(self, node: MENTSNode[TState, TAction]):
        self.simulate_action(node)
        while True: 
            action = self.e2w(node)
            if self.verbose:
                print("e2w chosen action:", action.value)
            observation, node_reward, terminated, truncated, _ = self.env.step(action.value)
            node.reward[action.value] = node_reward
            if self.verbose:
                print("instant reward:", node_reward)
                for child in node.children:
                    print(f"Checking child: {child}")
            child_node = next((child for child in node.children if child.inducing_action.value == action.value), None)
            if self.verbose:
                print("child_node:", child_node)
                print("node children", node.children)
            if child_node:
                # Continue looping with this child node
                node = child_node
            else:
                # If no such child exists, return the current node
                return node, action

    def expand(self, node: MENTSNode[TState, TAction], action):
        new_node = MENTSNode(node, action)
        node.add_child(new_node)
        self.simulate_action(new_node)
        self.reset_node(new_node)
        new_node.n += 1
        self.env.unwrapped.restore_state(node.state.current_state)
        observation, int_reward, terminated, truncated, _ = self.env.step(action.value)
        node.reward[action.value] = int_reward
        return new_node


    def simulate(self, node):
        current_state = self.env.unwrapped.clone_state(include_rng=True)
        parent_node = node.parent
        done = False
        parent_node.future_reward = 0
        while not done:
            random_action = random.choice(node.valid_actions)
            observation, r_reward, terminated, truncated, _ = self.env.step(random_action.value)
            done = terminated or truncated
            parent_node.future_reward += r_reward
            if done:
                self.env.unwrapped.restore_state(current_state)
                if self.verbose:
                    print("future reward:",parent_node.future_reward)
                break

    def backpropagate(self, node, action):
        node.n += 1
        node.Q_sft[action.value] = node.reward[action.value] + node.future_reward
        parent_node = node.parent

        while parent_node: 
            # Softmax backup
            node.n += 1
            for action in node.valid_actions:
                parent_node.Q_sft[action.value] = node.reward[action.value] + self.softmax_value(node.Q_sft)
            if self.verbose:
                print("Q_sft:", node.Q_sft)
            node = parent_node
            parent_node = parent_node.parent 








        