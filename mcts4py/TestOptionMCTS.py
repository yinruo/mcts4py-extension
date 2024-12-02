import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
import matplotlib.pyplot as plt
from samples.option.TestUSoptionMDP import TestUSoptionAction
import matplotlib.pyplot as plt
from longstaff_schwartz.algorithm import longstaff_schwartz
from mcts4py.SolverOptionMCTS import SolverOption
from samples.option.USoptionMDP import USoptionMDPOG
class TestOptionMCTS(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

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
        all_asset_prices = []  # List to store asset price paths from each run
        num_runs = 10          # Number of simulations to run

        for run in range(num_runs):
            print("run ",run)
            asset_prices = []
            root_node = ActionNode[TState, TAction](None, None)
            self.simulate_action(root_node)
            self.reset_tree(root_node)
            current_node = root_node
            while current_node.state.is_terminal == False:
                mdp = USoptionMDPOG(option_type=self.mdp.option_type, S0=self.mdp.S0, K=self.mdp.K, r=self.mdp.r, T=self.mdp.T, dt=self.mdp.dt, sigma=self.mdp.sigma)
                US_solver = SolverOption(
                    mdp,
                    simulation_depth_limit=100,
                    exploration_constant=1.0,
                    verbose=False
                )
                mcts_price = US_solver.run_option()
                intrinsic_price = self.get_payoff(current_node.state.asset_price)
                if self.verbose:
                    print("mcts_price",mcts_price)
                    print("intrinsic_price:",intrinsic_price )
                if intrinsic_price > mcts_price: 
                    #execute
                    action_taken = TestUSoptionAction.EXERCISE
                    new_node = ActionNode(current_node, action_taken)
                    current_node.add_child(new_node)
                    self.simulate_action(new_node)
                    print("get profit:" ,intrinsic_price)
                    break
                else:
                    action_taken = TestUSoptionAction.HOLD
                    new_node = ActionNode(current_node, action_taken)
                    current_node.add_child(new_node)
                    self.simulate_action(new_node)   
                    current_node = new_node
                if self.verbose:     
                    print("current asset_price: ", current_node.state.asset_price)
                    print("current time step:", current_node.state.time_step)
                asset_prices.append(current_node.state.asset_price)
            all_asset_prices.append(asset_prices)
        # Now plot all the asset price lines
        plt.figure(figsize=(10, 6))
        for i, asset_prices in enumerate(all_asset_prices):
            time_steps = list(range(len(asset_prices)))
            plt.plot(time_steps, asset_prices, marker='o', linestyle='-', label=f'Run {i+1}')
        plt.title('Asset Price over Time for Multiple Simulations')
        plt.xlabel('Time Steps')
        plt.ylabel('Asset Price')
        plt.legend()
        plt.grid(True)
        plt.show()
        return root_node.reward
    
    def get_payoff(self, S):
        if self.mdp.option_type == "Put":
            return np.maximum(self.mdp.K - S, 0)  
        elif self.mdp.option_type == "Call":
            return np.maximum(S - self.mdp.K, 0)

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

            # This state has been explored, select best action
            current_node = max(current_children, key=lambda c: self.calculate_uct(c))
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
        self.env.unwrapped.restore_state(node.state.current_state)
        valid_actions = self.mdp.actions()
        total_reward = 0
        done = False
        while not done:
            random_action = random.choice(valid_actions)
            observation, reward, terminated, truncated, _ = self.env.step(random_action.value)
            done = terminated or truncated
            total_reward += reward
            if done:
                self.env.unwrapped.restore_state(node.state.current_state)
                break
        return total_reward 

    def backpropagate(self, node: ActionNode[TState, TAction], reward: float) -> None:
        current_node = node
        current_reward = reward

        while current_node != None:
            current_node.max_reward = max(current_reward, current_node.max_reward)
            current_node.reward += current_reward
            current_node.n += 1

            current_node = current_node.parent
            current_reward *= self.discount_factor
