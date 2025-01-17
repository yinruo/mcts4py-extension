import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
import matplotlib.pyplot as plt
from samples.option.USoptionMDP import USoptionAction
import matplotlib.pyplot as plt
class OptionSolver(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):
    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 vc:bool,
                 verbose: bool = False):
        self.vc = vc
        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(self.root_node)

        super().__init__(exploration_constant, verbose)   

    def root(self) -> ActionNode[TState, TAction]:
        return self.root_node 

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

    def get_intrinsic_value(self, S):
        if self.mdp.option_type == "Put":
            return np.maximum(self.mdp.K - S, 0)  
        elif self.mdp.option_type == "Call":
            return np.maximum(S - self.mdp.K, 0)
        
    def detach_parent(self,node: ActionNode[TState, TAction]):
        del node.parent
        node.parent = None

    def run_option(self):
        root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        while True:    
            root_node,action = self.run_iteration(root_node, 100)
            #self.print_asset_price_tree(root_node)
            #new_node = ActionNode(current_node, action)
            #current_node.add_child(new_node)
            #self.simulate_action(new_node)
            if action == USoptionAction.EXERCISE:
                #print("the action is exercise")
                final_node = root_node
                intrinsic_value = self.mdp.get_intrinsic_value(final_node.state.asset_price)
                if self.verbose:
                    print("the final reward is", intrinsic_value)
                return intrinsic_value
                #print("the action is hold")
            if root_node.state.time_step == self.mdp.T:
                intrinsic_value = self.mdp.get_intrinsic_value(root_node.state.asset_price)
                if self.verbose:
                    print("the final reward is", intrinsic_value)
                return intrinsic_value
            self.detach_parent(root_node)
    

    def run_option_hindsight(self):
        root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        while True:    
            root_node,action = self.run_iteration_hindsight(root_node, 400)
            if action == USoptionAction.EXERCISE:
                #final_node = root_node.parent
                intrinsic_value = self.mdp.get_intrinsic_value(root_node.state.asset_price)
                if self.verbose:
                    print("the final reward is", intrinsic_value)
                return intrinsic_value
            if root_node.state.time_step == self.mdp.T:
                intrinsic_value = self.mdp.get_intrinsic_value(root_node.state.asset_price)
                if self.verbose:
                    print("the final reward is", intrinsic_value)
                return intrinsic_value
            self.detach_parent(root_node)

    def run_baseline(self):
        root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        current_state = root_node.state
        while True:
            action = USoptionAction.HOLD
            new_state = self.mdp.transition(current_state, action)
            if new_state.time_step == self.mdp.T or new_state.is_terminal == True:
                intrinsic_value = self.mdp.get_intrinsic_value(current_state.asset_price)
                #print("reward for this round",intrinsic_value)
                return intrinsic_value
            current_state = new_state
        

    def run_iteration(self, node: ActionNode[TState, TAction],iterations:int):
        for i in range(iterations):
            explore_node = self.select(node)
            expanded_node = self.expand(explore_node)
            if self.vc:
                self.simulate_hindsight(expanded_node)
            else:
                self.simulate(expanded_node)
            self.backpropagate(expanded_node)
        next_node,next_action = self.next(node)
        return next_node,next_action
    


    def run_iteration_hindsight(self, node: ActionNode[TState, TAction],iterations:int):
        for i in range(iterations):
            explore_node = self.select(node)
            expanded_node = self.expand(explore_node)
            self.simulate_hindsight(expanded_node)
            self.backpropagate(expanded_node)
        next_node,next_action = self.next_hindsight(node)
        return next_node,next_action

    def next(self,node: ActionNode[TState, TAction]):

        if self.mdp.is_terminal(node.state):
            raise ValueError("Option has ended")

        children = node.children
        max_n = max(node.n for node in children)
        #print("number of visits" ,max_n)

        best_children = [c for c in children if c.n == max_n]
        best_child = random.choice(best_children)

        return best_child, best_child.inducing_action
    
    def next_hindsight(self,node: ActionNode[TState, TAction]):

        if self.mdp.is_terminal(node.state):
            raise ValueError("Option has ended")

        children = node.children
        max_r = max(node.reward for node in children)
        #print("number of visits" ,max_n)

        best_children = [c for c in children if c.reward == max_r]
        best_child = random.choice(best_children)

        return best_child, best_child.inducing_action

    def select(self, node: ActionNode[TState, TAction], iteration_number=None) -> ActionNode[TState, TAction]:
        if len(node.children) == 0:
            return node

        current_node = node

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

    def expand(self, node: ActionNode[TState, TAction], iteration_number=None) -> ActionNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node
        
        valid_actions = node.valid_actions

        if np.random.rand() < self.mdp.p:
            price_up = True
        else:
            price_up = False

        new_nodes = []

        for action in valid_actions:
            new_node = ActionNode(node, action)
            node.add_child(new_node)
            self.simulate_action(new_node)
            new_nodes.append(new_node)
        
        expand_node = random.choice(new_nodes)
        return expand_node
    
    def simulate(self, node: ActionNode[TState, TAction], depth=0) -> float:
        if node.inducing_action == USoptionAction.EXERCISE:
            node = node.parent
            intrinsic_value = self.mdp.get_intrinsic_value(node.state.asset_price)
            node.reward = intrinsic_value
        else:
            current_state = node.state
            while True:
                action = USoptionAction.HOLD
                #actions = self.mdp.actions(node.state)
                #random_action = random.choice(actions)
                new_state = self.mdp.transition(current_state, action)
                intrinsic_value = self.mdp.get_intrinsic_value(new_state.asset_price)
                if new_state.time_step == self.mdp.T or new_state.is_terminal == True or intrinsic_value > 0:
                    node.reward = intrinsic_value
                    break
                current_state = new_state

    def simulate_hindsight(self, node: ActionNode[TState, TAction], depth=0) -> float:
        current_state = node.state
        asset_prices = [current_state.asset_price]
        time = current_state.time_step
        while time < self.mdp.T:
            action = USoptionAction.HOLD
            new_state = self.mdp.transition(current_state, action)
            asset_prices.append(new_state.asset_price)
            current_state = new_state
            time = current_state.time_step

        intrinsic_values = []
        for asset_price in asset_prices:
            intrinsic_value = self.mdp.get_intrinsic_value(asset_price)
            intrinsic_values.append(intrinsic_value)
        max_payoff = max(intrinsic_values)
        node.reward = max_payoff

    def backpropagate(self, node: ActionNode[TState, TAction]) -> None:
        current_node = node
        current_reward = current_node.reward
        current_node.n += 1
        current_node = current_node.parent
        while current_node != None:
            time_decay = 1 - (current_node.state.time_step / self.mdp.T) 
            current_reward *= time_decay 
            current_node.reward += current_reward
            current_node.n += 1
            current_node = current_node.parent
            