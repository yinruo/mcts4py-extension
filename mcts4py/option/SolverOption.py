import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
import gymnasium as gym
from samples.option.test.USoptionMDP import USoptionAction, USoptionState
import matplotlib.pyplot as plt
from longstaff_schwartz.algorithm import longstaff_schwartz
from longstaff_schwartz.stochastic_process import GeometricBrownianMotion
from samples.option.test.bino import BinomialTreeOption
from blackscholes import BlackScholesPut, BlackScholesCall
class SolverOption(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 discount_factor: float,
                 verbose: bool = False):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        self.__root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(self.__root_node)

        super().__init__(exploration_constant, verbose)

    def root(self) -> ActionNode[TState, TAction]:
        return self.__root_node
    
    def run_option(self):
        root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        self.reset_tree(root_node)
        self.run_iteration(root_node, 100000)
        self.print_asset_price_tree(root_node)
        print("The price of the american option is",root_node.reward)
        return root_node.reward
    
    def run_iteration(self, node: ActionNode[TState, TAction],iterations:int):
        for i in range(iterations):
            explore_node = self.select(node)
            expanded = self.expand(explore_node)
            self.simulate(expanded)
            self.backpropagate(expanded)
            #simulated_reward = self.simulate(expanded)
            #self.backpropagate(expanded, simulated_reward) 
            all_terminal = self.all_leaf_nodes_terminal(node)
            if all_terminal: 
                break
    
    def get_payoff(self, S):
        if self.mdp.option_type == "Put":
            return np.maximum(self.mdp.K - S, 0)  
        elif self.mdp.option_type == "Call":
            return np.maximum(S - self.mdp.K, 0)
          
    def simulate_gbm(self, S, dt, r, sigma):
        Z = np.random.normal()
        return S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    """     def simulate_future(self, S):
        future_price = S
        current_time = 0

        while current_time < self.mdp.T:
            future_price = self.simulate_gbm(future_price, self.mdp.dt, self.mdp.r, self.mdp.sigma)
            current_time += self.mdp.dt

        final_payoff = self.get_payoff(future_price)
        discounted_payoff = final_payoff * np.exp(-self.mdp.r * self.mdp.T)
        
        return discounted_payoff """

    def simulate_future(self, S):
        #if self.mdp.option_type == "Put":
            #putCall = BlackScholesPut(S = S, K=self.mdp.K, T=self.mdp.T, r=self.mdp.r, sigma= self.mdp.sigma, q=0)
            #binomial_price = putCall.price() 
        #else:
            #callCall = BlackScholesCall(S = S, K=self.mdp.K, T=self.mdp.T, r=self.mdp.r, sigma= self.mdp.sigma, q=0)
            #binomial_price = callCall.price() 
        binomial_model = BinomialTreeOption(S, self.mdp.K, self.mdp.r, self.mdp.T, self.mdp.sigma, self.mdp.dt, self.mdp.option_type)
        binomial_price = binomial_model.price()
        return binomial_price


    def select(self, node: ActionNode[TState, TAction], iteration_number=None) -> ActionNode[TState, TAction]:
        current_node = node
        if len(current_node.children) == 0:
            return current_node

        while True:
            current_children = current_node.children
            # If the node is terminal, return it
            if self.mdp.is_terminal(current_node.state) or len(current_children)==0:
                return current_node

            
            if np.random.rand() < self.mdp.p:
                children_up = [child for child in current_children if child.inducing_action == USoptionAction.UP]
                if children_up:
                    current_node = random.choice(children_up)
                else:
                    raise ValueError("No child with UP action found")
            else:
                children_down = [child for child in current_children if child.inducing_action == USoptionAction.DOWN]
                if children_down:
                    current_node = random.choice(children_down)
                else:
                    raise ValueError("No child with DOWN action found")

    def expand(self, node: ActionNode[TState, TAction], iteration_number=None) -> ActionNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node
        valid_actions = self.mdp.actions(node.state)

        for action in valid_actions:
            new_node = ActionNode(node, action)
            node.add_child(new_node)
            self.simulate_action(new_node)

        
        return random.choice(node.children)

    def simulate(self, node: ActionNode[TState, TAction], depth=0) -> float:
        asset_price = node.state.asset_price
        time_step = node.state.time_step

        immediate_payoff = self.get_payoff(asset_price)
        node.state.imme = immediate_payoff
        future_expected_reward = self.simulate_future(asset_price)
        node.state.expect = future_expected_reward

        if future_expected_reward > immediate_payoff:
            node.reward = future_expected_reward
            return "HOLD"
        else:
            node.reward = immediate_payoff
            node.state.is_terminal = True
            return "EXERCISE"
            

    def backpropagate(self, node: ActionNode[TState, TAction]) -> None:
        node.n += 1
        current_node = node.parent
        while current_node is not None:
            current_node.n += 1
            future_rewards = [child.reward for child in current_node.children if child.reward is not None]
            immediate_payoff = self.get_payoff(current_node.state.asset_price)
            # If children exist, calculate the average reward
            if future_rewards:
                average_reward = np.mean(future_rewards)  # Calculate the average reward of the children
                if average_reward > immediate_payoff:
                    current_node.reward = average_reward
                else:
                    current_node.reward = immediate_payoff
            else:
                current_node.reward = immediate_payoff
            current_node = current_node.parent

    # Utilities

    def simulate_action(self, node: ActionNode[TState, TAction]):
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

    def reset_tree(self, node: ActionNode[TState, TAction]):
        # Reset the node's statistics
        node.n = 0
        self.reward = 0.0
        self.max_reward = 0.0
        node.__children = []


    def print_tree(self, node: ActionNode[TState, TAction], level=0, is_last=True, prefix=""):
        """Recursive tree printing, showing both asset price and option price at each node"""
        indent = prefix + ("|-- " if is_last else "|-- ")
        
        # Print both asset price and option price at each node
        print(f"{indent}asset price: {node.state.asset_price:.4f}, option price: {node.reward:.4f}, time: {node.state.time_step} ")

        prefix += "    " if is_last else "|   "
        
        # Recursively print children nodes
        num_children = len(node.children)
        for idx, child in enumerate(node.children):
            is_last_child = (idx == num_children - 1)
            self.print_tree(child, level + 1, is_last_child, prefix)

    def print_asset_price_tree(self, node):
        print("资产价格树：")
        self.print_tree(node)

    def all_leaf_nodes_terminal(self,node):
        if not node.children:
            return node.state.is_terminal
        return all(self.all_leaf_nodes_terminal(child) for child in node.children)
        



