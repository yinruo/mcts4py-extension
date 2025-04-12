import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
import matplotlib.pyplot as plt
from samples.option.Multi.USMultiOptionMDP import USMultiOptionMDP,MultiOptionAction
import matplotlib.pyplot as plt
class MultiOptionSolver(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):
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

        new_nodes = []

        for action in valid_actions:
            new_node = ActionNode(node, action)
            node.add_child(new_node)
            self.simulate_action(new_node)
            new_nodes.append(new_node)
        
        expand_node = random.choice(new_nodes)
        return expand_node

    
    def simulate(self, node: ActionNode[TState, TAction]) -> float:
        current_state = node.state
        exercised_flags = current_state.exercised_flags[:]
        reward = 0.0
        T = self.mdp.T
        n = len(current_state.asset_prices)

        while not self.mdp.is_terminal(current_state):
            valid_actions = self.mdp.actions(current_state)
            action = random.choice(valid_actions)

            next_state = self.mdp.transition(current_state, action)

            for i in range(n):
                if not exercised_flags[i] and next_state.exercised_flags[i]:
                    price = next_state.asset_prices[i]
                    K = self.mdp.K_list[i]
                    opt_type = self.mdp.option_type_list[i]
                    val = self.mdp.get_intrinsic_value(price, K, opt_type)
                    reward += val
                    exercised_flags[i] = True

            current_state = next_state

        for i in range(n):
            if not exercised_flags[i]:
                price = current_state.asset_prices[i]
                val = self.mdp.get_intrinsic_value(price, self.mdp.K_list[i], self.mdp.option_type_list[i])
                if val > 0:
                    reward += val

        node.reward = reward
        return reward



    def simulate_hindsight(self, node: ActionNode[TState, TAction]) -> float:
        current_state = node.state
        T = self.mdp.T
        time = current_state.time_step
        n = len(current_state.asset_prices)

        price_paths = []
        while time < T and not self.mdp.is_terminal(current_state):
            hold_action = MultiOptionAction([0] * n)
            next_state = self.mdp.transition(current_state, hold_action)
            #print(f"[simulate_hindsight] transitioning from time {current_state.time_step} to {next_state.time_step}, terminal: {self.mdp.is_terminal(next_state)}")
            time = next_state.time_step
            #print("current time step", state.time_step)
            #print("next time step", next_state.time_step)
            price_paths.append(next_state)
            current_state = next_state
            #print("time,",time)
        
        reward = 0.0
        for i in range(n):
            best_val = 0.0
            for state in price_paths:
                val = self.mdp.get_intrinsic_value(
                    state.asset_prices[i], self.mdp.K_list[i], self.mdp.option_type_list[i])
                if val > best_val:
                    best_val = val
            reward += best_val 

        node.reward = reward
        return reward
        


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
            
    def simulate_action(self, node: ActionNode[TState, TAction]):
        if node.parent == None:
            initial_state = self.mdp.initial_state()
            node.state = initial_state
            node.valid_actions = self.mdp.actions(node.state)
            return

        if node.inducing_action == None:
            raise RuntimeError("Action was null for non-null parent")
        new_state = self.mdp.transition(node.parent.state, node.inducing_action)
        #print("UCT, generated asset prices for new state,", new_state.asset_prices)
        node.state = new_state
        node.valid_actions = self.mdp.actions(node.state)
        
    def detach_parent(self,node: ActionNode[TState, TAction]):
        del node.parent
        node.parent = None

    def run_iteration(self, node: ActionNode[TState, TAction],iterations:int):
        if self.mdp.is_terminal(node.state):
            return node, None
        for i in range(iterations):
            explore_node = self.select(node)
            expanded_node = self.expand(explore_node)
            self.simulate(expanded_node)
            self.backpropagate(expanded_node)
        next_node,next_action = self.next(node)
        #print("UCT, next_node 's asset prices", next_node.state.asset_prices)
        return next_node,next_action
    
    def run_iteration_hindsight(self, node: ActionNode[TState, TAction],iterations:int):
        if self.mdp.is_terminal(node.state):
            return node, None
        for i in range(iterations):
            explore_node = self.select(node)
            expanded_node = self.expand(explore_node)
            self.simulate_hindsight(expanded_node)
            self.backpropagate(expanded_node)

        next_node,next_action = self.next(node)
        return next_node,next_action

    def next(self,node: ActionNode[TState, TAction]):

        if self.mdp.is_terminal(node.state):
            raise ValueError("Option has ended")

        children = node.children
        max_n = max(node.n for node in children)

        best_children = [c for c in children if c.n == max_n]
        best_child = random.choice(best_children)

        return best_child, best_child.inducing_action


    def run_option(self):
        root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        #print("UCT, inital asset price,", root_node.state.asset_prices)
        reward = 0.0

        while True:
            root_node, action = self.run_iteration(root_node, 400)

            print("method 1, action is", action.decisions)

            prev_flags = root_node.parent.state.exercised_flags[:]
            #print("method 1, old flags, ",prev_flags)
            prev_state = root_node.parent.state
            next_state = self.mdp.transition(prev_state, action)
            #next_state = root_node.state
            #root_node.valid_actions = self.mdp.actions(next_state)
            new_flags = next_state.exercised_flags
            #print("method 1, new flags, ",new_flags)
            exercised_now = [
                i for i, (prev, new) in enumerate(zip(prev_flags, new_flags))
                if not prev and new
            ]
            #print("method 1, exercised_now", exercised_now)
            #print("method 1, exercised_now", exercised_now)
            #print("method 1, next state accet prices",next_state.asset_prices )
            for i in exercised_now:
                price = next_state.asset_prices[i]
                K = self.mdp.K_list[i]
                opt_type = self.mdp.option_type_list[i]
                val = self.mdp.get_intrinsic_value(price, K, opt_type)
                #print("method 1, val is added to the reward, val=", val)
                reward += val

            if all(next_state.exercised_flags):
                #print("method 1, final reward", reward)
                return reward
            
            
            if next_state.time_step == self.mdp.T:
                for i, was_exercised in enumerate(next_state.exercised_flags):
                    if not was_exercised:
                        price = next_state.asset_prices[i]
                        #print("method 1, price", price)
                        K = self.mdp.K_list[i]
                        opt_type = self.mdp.option_type_list[i]
                        val = self.mdp.get_intrinsic_value(price, K, opt_type)
                        if val > 0:
                            #print("method 1, val is added to the reward at terminal state, val=", val)
                            reward += val
                #print("method 1, final reward", reward)
                return reward
            root_node.state = next_state
            #print("next state time step", next_state.time_step)


    def run_option_hindsight(self):
        root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        reward = 0.0

        while True:
            root_node, action = self.run_iteration_hindsight(root_node, 400)
            prev_flags = root_node.parent.state.exercised_flags[:]
            prev_state = root_node.parent.state
            next_state = self.mdp.transition(prev_state, action)
            exercised_now = [
                i for i, (prev, new) in enumerate(zip(prev_flags, next_state.exercised_flags))
                if not prev and new
            ]
            print("method 2, next state accet prices",next_state.asset_prices )
            for i in exercised_now:
                price = next_state.asset_prices[i]
                K = self.mdp.K_list[i]
                opt_type = self.mdp.option_type_list[i]
                val = self.mdp.get_intrinsic_value(price, K, opt_type)
                reward += val

            if all(next_state.exercised_flags):
                return reward

            if next_state.time_step == self.mdp.T:
                for i, was_exercised in enumerate(next_state.exercised_flags):
                    if not was_exercised:
                        price = next_state.asset_prices[i]
                        K = self.mdp.K_list[i]
                        opt_type = self.mdp.option_type_list[i]
                        val = self.mdp.get_intrinsic_value(price, K, opt_type)
                        if val > 0:
                            reward += val
                return reward
            root_node.state = next_state
