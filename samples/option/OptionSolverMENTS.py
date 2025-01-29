import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
from samples.option.USoptionMDP import USoptionAction

class OptionSolverMENTS(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):
    def __init__(self,
                 mdp: MDP[TState, TAction],
                 exploration_constant: float,
                 discount_factor: float,
                 temperature:float,
                 epsilon:float,
                 verbose: bool = False):

        self.mdp = mdp
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.epsilon = epsilon
        self.root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(self.root_node)

        super().__init__(exploration_constant, verbose)   

    def root(self) -> MENTSNode[TState, TAction]:
        return self.root_node 
    
    def e2w(self, node: MENTSNode[TState, TAction]):
        total_visits = 0
        for action in node.valid_actions:
            total_visits += node.visits[action.value]
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

        #print("Number of valid_actions:", len(node.valid_actions))
        #print("Number of action_probabilities:", len(action_probabilities))
        #print("Q_sft keys:", list(node.Q_sft.keys()))
        action = np.random.choice(node.valid_actions, p=action_probabilities)
        return action
    

    def softmax_value(self, Q_values):
        Q_values_array = np.array(list(Q_values.values()))
        max_Q = np.max(Q_values_array) 
        exp_values = np.exp((Q_values_array - max_Q) / self.temperature) 
        sum_exp_values = np.sum(exp_values)  
        softmax_val = self.temperature * np.log(sum_exp_values)+ max_Q  
        return softmax_val
    
    def soft_indmax(self, Q_values):
        softmax = self.softmax_value(Q_values)
        Q_values_array = np.array(list(Q_values.values()))
        # Calculate fτ(r) using the formula exp((r - Fτ(r)) / τ)
        soft_indmax_value = np.exp((Q_values_array - softmax) / self.temperature)
        soft_indmax_value /= np.sum(soft_indmax_value)
        return soft_indmax_value

    def simulate_action(self, node: MENTSNode[TState, TAction]):
        if node.parent == None:
            initial_state = self.mdp.initial_state()
            node.state = initial_state
            node.valid_actions = self.mdp.actions(node.state)
            for action in node.valid_actions:
                node.Q_sft[action.value] = 0.0 
                node.action_reward[action.value] = 0.0
            return

        if node.inducing_action == None:
            raise RuntimeError("Action was null for non-null parent")
        new_state = self.mdp.transition(node.parent.state, node.inducing_action)
        node.state = new_state
        node.valid_actions = self.mdp.actions(node.state)   
        for action in node.valid_actions:
            node.Q_sft[action.value] = 0.0 

    def get_intrinsic_value(self, S):
        if self.mdp.option_type == "Put":
            return np.maximum(self.mdp.K - S, 0)  
        elif self.mdp.option_type == "Call":
            return np.maximum(S - self.mdp.K, 0)
        
    def detach_parent(self,node: MENTSNode[TState, TAction]):
        del node.parent
        node.parent = None

    def run_option(self):
        root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        while True:    
            root_node,action,rewards = self.run_iteration(root_node, 200)
            if action == USoptionAction.EXERCISE:
                root_node = root_node.parent
                intrinsic_value = self.mdp.get_intrinsic_value(root_node.state.asset_price)
                return intrinsic_value
            if root_node.state.time_step == self.mdp.T:
                intrinsic_value = self.mdp.get_intrinsic_value(root_node.state.asset_price)
                if self.verbose:
                    print("the final reward is", intrinsic_value)
                return intrinsic_value
            self.detach_parent(root_node)
    

    def run_option_hindsight(self):
        root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        while True:    
            root_node,action,rewards = self.run_iteration_hindsight(root_node, 200)
            #self.print_asset_price_tree(root_node)
            #new_node = MENTSNode(current_node, action)
            #current_node.add_child(new_node)
            #self.simulate_action(new_node)
            if action == USoptionAction.EXERCISE:
                intrinsic_value = self.mdp.get_intrinsic_value(root_node.state.asset_price)
                return intrinsic_value
            if root_node.state.time_step == self.mdp.T:
                intrinsic_value = self.mdp.get_intrinsic_value(root_node.state.asset_price)
                return intrinsic_value
            self.detach_parent(root_node)

    def get_root_rewards(self):
        root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        root_node,action_taken, root_rewards = self.run_iteration(root_node, 200)
        return root_rewards
    
    def get_root_rewards_hindsight(self):
        root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        root_node,action_taken, root_rewards = self.run_iteration_hindsight(root_node, 200)
        return root_rewards
    
    def run_baseline(self):
        root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        current_state = root_node.state
        while True:
            action = USoptionAction.HOLD
            new_state = self.mdp.transition(current_state, action)
            if new_state.time_step == self.mdp.T or new_state.is_terminal == True:
                intrinsic_value = self.mdp.get_intrinsic_value(current_state.asset_price)
                if self.verbose:
                    print("reward for this round",intrinsic_value)
                return intrinsic_value
            current_state = new_state
        

    def run_iteration(self, node: MENTSNode[TState, TAction],iterations:int):
        root_rewards = []
        for i in range(iterations):
            explore_node = self.select(node)
            if self.verbose:
                print("explore node qsft", explore_node.Q_sft)
                print("select end, expand start")
            expanded_node,action_taken = self.expand(explore_node)
            if self.verbose:
                print("expand node qsft", expanded_node.Q_sft)
                print("expand end, simulate start")
            simulated_reward = self.simulate(expanded_node)
            if self.verbose:
                print("simulate end, backpropagate start")
            self.backpropagate(expanded_node,action_taken, simulated_reward)
            if self.verbose:
                print("backpropagate end, select start")
            #print(node.reward)
            root_rewards.append(node.reward)
        next_node,next_action = self.next(node)
        return next_node,next_action,root_rewards
    
    def run_iteration_hindsight(self, node: MENTSNode[TState, TAction],iterations:int):
        root_rewards = []
        for i in range(iterations):
            explore_node = self.select(node)
            expanded_node,action_taken = self.expand(explore_node)
            simulated_reward =self.simulate_hindsight(expanded_node)
            self.backpropagate(expanded_node,action_taken, simulated_reward)
            print(node.reward)
            root_rewards.append(node.reward)
        next_node,next_action = self.next(node)
        return next_node,next_action,root_rewards

    def next(self,node: MENTSNode[TState, TAction]):
        if self.mdp.is_terminal(node.state):
            raise ValueError("Option has ended")
        
        soft_indmax_probs = self.soft_indmax(node.Q_sft)
        print(soft_indmax_probs)
        index_of_better_value = np.argmax(soft_indmax_probs)
        best_child = None
        for child in node.children:
            if child.inducing_action.value == index_of_better_value:
                best_child = child
                break

        output_list = []
        for action in node.valid_actions:
            # node.visits[action.value] holds the visit count
            output_list.append(f"{action.value}:{node.visits[action.value]}")

        # Create a string like [1:3, 2:50, 3:10]
        output_str = "[" + ", ".join(output_list) + "]"

        print(output_str)
        return best_child, best_child.inducing_action
    

    def select(self, node: MENTSNode[TState, TAction], iteration_number=None) -> MENTSNode[TState, TAction]:
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
            if self.verbose:
                print(" current node ,", current_node.Q_sft)
            action = self.e2w(current_node)
            current_node = next((child for child in current_children if child.inducing_action == action), None)

    def expand(self, node: MENTSNode[TState, TAction], iteration_number=None) -> MENTSNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node, node.inducing_action
        
        valid_actions = node.valid_actions
        random_action = random.choice(valid_actions)
        new_node = MENTSNode(node, random_action)
        node.add_child(new_node)
        self.simulate_action(new_node)
        
        return new_node, new_node.inducing_action
    
    def simulate(self, node: MENTSNode[TState, TAction], depth=0) -> float:
        if node.inducing_action == USoptionAction.EXERCISE:
            intrinsic_value = self.mdp.get_intrinsic_value(node.state.asset_price)
            return intrinsic_value
        else:
            current_state = node.state
            while True:
                action = USoptionAction.HOLD
                #actions = self.mdp.actions(node.state)
                #random_action = random.choice(actions)
                new_state = self.mdp.transition(current_state, action)
                intrinsic_value = self.mdp.get_intrinsic_value(new_state.asset_price)
                if new_state.time_step == self.mdp.T or new_state.is_terminal == True or intrinsic_value > 0:
                    return intrinsic_value
                current_state = new_state

    def simulate_hindsight(self, node: MENTSNode[TState, TAction], depth=0) -> float:
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
        return max_payoff

    def backpropagate(self, node: MENTSNode[TState, TAction],action, reward: float) -> None:
        current_reward = reward
        node.visits[action.value] += 1
        node.Q_sft[action.value] = node.action_reward[action.value] + reward
        node.reward += current_reward
        #print("reward:", node.reward)
        softmax_value = self.softmax_value(node.Q_sft)
        inducing_action = node.inducing_action 
        node = node.parent 

        while node:
            node.visits[inducing_action.value] += 1
            node.Q_sft[inducing_action.value] = node.action_reward[inducing_action.value] + softmax_value
            node.reward += current_reward
            #print("reward:", node.reward)
            if self.verbose:
                print("softmax value:", softmax_value)
                print("Q_sft:", node.Q_sft)

            softmax_value = self.softmax_value(node.Q_sft)
            inducing_action = node.inducing_action  
            node = node.parent 
            current_reward *= self.discount_factor


    def get_root_rewards(self):
        root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        _,_, root_rewards = self.run_iteration(root_node, 200)
        return root_rewards
    
    def get_root_rewards_hindsight(self):
        root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        _,_, root_rewards = self.run_iteration_hindsight(root_node, 200)
        return root_rewards
            
            