import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
import gymnasium as gym
from samples.option.USoptionMDP import USoptionAction, USoptionState
import matplotlib.pyplot as plt
from longstaff_schwartz.algorithm import longstaff_schwartz
from longstaff_schwartz.stochastic_process import GeometricBrownianMotion
from samples.option.test.bino import BinomialTreeOption
from blackscholes import BlackScholesPut, BlackScholesCall
class SolverOptionMENTS(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 verbose: bool = False):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.__root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(self.__root_node)

        super().__init__(exploration_constant, verbose)

    def root(self) -> ActionNode[TState, TAction]:
        return self.__root_node
    
    def run_option(self):
        root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        self.reset_tree(root_node)
        self.run_iteration(root_node, 100)
        self.print_asset_price_tree(root_node)
        root_node.reward = round(root_node.reward, 4)
        print("The price of the american option is",root_node.reward)
        return root_node.reward
    
    def run_iteration(self, node: ActionNode[TState, TAction],iterations:int):
        for i in range(iterations):
            explore_node = self.select(node)
            expanded = self.expand(explore_node)
            if expanded.inducing_action == USoptionAction.UP_HOLD or expanded.inducing_action == USoptionAction.DOWN_HOLD:
                self.simulate(expanded)
            else: 
                expanded.reward = self.get_payoff(expanded.state.asset_price)
            self.backpropagate(expanded)
            #simulated_reward = self.simulate(expanded)
            #self.backpropagate(expanded, simulated_reward) 
            #all_terminal = self.all_leaf_nodes_terminal(node)
            #if all_terminal: 
                #break
    
    def get_payoff(self, S):
        if self.mdp.option_type == "Put":
            return np.maximum(self.mdp.K - S, 0)  
        elif self.mdp.option_type == "Call":
            return np.maximum(S - self.mdp.K, 0)
          
    """ def simulate_gbm(self, S, dt, r, sigma):
        Z = np.random.normal()
        return S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z) """

    """     def simulate_future(self, S):
        future_price = S
        current_time = 0

        while current_time < self.mdp.T:
            future_price = self.simulate_gbm(future_price, self.mdp.dt, self.mdp.r, self.mdp.sigma)
            current_time += self.mdp.dt

        final_payoff = self.get_payoff(future_price)
        discounted_payoff = final_payoff * np.exp(-self.mdp.r * self.mdp.T)
        
        return discounted_payoff """

    """ def simulate_future(self, S):
        #if self.mdp.option_type == "Put":
            #putCall = BlackScholesPut(S = S, K=self.mdp.K, T=self.mdp.T, r=self.mdp.r, sigma= self.mdp.sigma, q=0)
            #binomial_price = putCall.price() 
        #else:
            #callCall = BlackScholesCall(S = S, K=self.mdp.K, T=self.mdp.T, r=self.mdp.r, sigma= self.mdp.sigma, q=0)
            #binomial_price = callCall.price() 
        binomial_model = BinomialTreeOption(S, self.mdp.K, self.mdp.r, self.mdp.T, self.mdp.sigma, self.mdp.dt, self.mdp.option_type)
        binomial_price = binomial_model.price()
        return binomial_price """


    def select(self, node: ActionNode[TState, TAction], iteration_number=None) -> ActionNode[TState, TAction]:
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
            
            exercise_children = [child for child in current_children if child.inducing_action == USoptionAction.UP_EXERCISE or child.inducing_action == USoptionAction.DOWN_EXERCISE]
            non_exercise_children = [child for child in current_children if child not in exercise_children]

            if non_exercise_children:
                current_node = max(non_exercise_children, key=lambda c: self.calculate_uct(c))
            else:
                print("No non-exercise children available")

            self.simulate_action(current_node)


    def expand(self, node: ActionNode[TState, TAction], iteration_number=None) -> ActionNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node

        current_children = node.children
        explored_actions = set([c.inducing_action for c in current_children])
        valid_action: set[TAction] = set(node.valid_actions)
        unexplored_actions = valid_action - explored_actions

        # Expand an unexplored action
        action_taken = random.sample(list(unexplored_actions), 1)[0]

        new_node = ActionNode(node, action_taken)
        node.add_child(new_node)
        self.simulate_action(new_node)

        return new_node

    def simulate(self, node: ActionNode[TState, TAction], depth=0) -> float:
        while True:
            actions = self.mdp.actions(node.state)
            random_action = random.choice(actions)
            new_state = self.mdp.transition(node.state, random_action)
            if new_state.time_step == self.mdp.T or new_state.is_terminal == True:
                final_reward = self.get_payoff(new_state.asset_price)
                node.reward = final_reward
                break
            

    def backpropagate(self, node: ActionNode[TState, TAction]) -> None:
        current_node = node
        current_reward = current_node.reward

        while current_node != None:
            current_node.reward += current_reward
            current_node.n += 1
            current_node = current_node.parent
            current_reward *= (1-self.mdp.r)

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
        print(f"{indent}asset price: {node.state.asset_price:.4f}, action: {node.inducing_action}, reward: {node.reward:.4f}, time: {node.n} ")

        prefix += "    " if is_last else "|   "
        
        # Recursively print children nodes
        num_children = len(node.children)
        for idx, child in enumerate(node.children):
            is_last_child = (idx == num_children - 1)
            self.print_tree(child, level + 1, is_last_child, prefix)

    def print_asset_price_tree(self, node):
        print("Tree：")
        node.reward /= node.n
        self.print_tree(node)

    def all_leaf_nodes_terminal(self,node):
        if not node.children:
            return node.state.is_terminal
        return all(self.all_leaf_nodes_terminal(child) for child in node.children)
        



