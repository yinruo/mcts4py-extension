import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
from samples.option.MENTS.QValueEstimator import *
from samples.option.USoptionMDP import USoptionAction
random = random.Random(0)
import inspect


accepts_arguments = lambda func, num_args: len(inspect.signature(func).parameters) == num_args


class StatefulSolverMENTS(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 discount_factor: float,
                 exploration_constant: float,
                 verbose: bool = False,
                 max_iteration: int = 1000,
                 early_stop: bool = False,
                 early_stop_condition: dict = None,
                 exploration_constant_decay = 1,
                 value_function_estimator_callback = None,
                 alpha_value = 0.5,
                 value_clipping: bool = False,
                 value_function_upper_estimator_callback = None,
                 value_function_lower_estimator_callback = None,
                 lambda_temp_callback=exponential_decay):
        self.mdp = mdp
        self.discount_factor = discount_factor
        self.simulation_depth_limit = simulation_depth_limit
        self.value_function_estimator_callback = value_function_estimator_callback
        self.alpha_value = alpha_value
        self.value_clipping = value_clipping
        self.value_function_upper_estimator_callback = value_function_upper_estimator_callback
        self.value_function_lower_estimator_callback = value_function_lower_estimator_callback

        self.q_estimator = QValueEstimator(alpha=alpha_value, lambda_temp_callback=lambda_temp_callback)

        self.ments_value_tracker = []
        
        super().__init__(exploration_constant, verbose, max_iteration,
                         early_stop, early_stop_condition, exploration_constant_decay)
        self.__root_node = self.create_node(None, None, mdp.initial_state())

    def root(self) -> StateNode[TState, TAction]:
        return self.__root_node
    
    def get_intrinsic_value(self, S):
        if self.mdp.option_type == "Put":
            return np.maximum(self.mdp.K - S, 0)  
        elif self.mdp.option_type == "Call":
            return np.maximum(S - self.mdp.K, 0)
        
    def detach_parent(self,node: MENTSNode[TState, TAction]):
        del node.parent
        node.parent = None
        
    def run_option(self):
        root_node = self.create_node(None, None, self.mdp.initial_state())
        while True:
            new_node, action_taken,node_reward = self.run_iteration(root_node, 300)
            #print("what is the parent node price",parent_node.state.asset_price)
            #new_node.state = self.mdp.transition(root_node.state, action_taken)
            if action_taken == USoptionAction.EXERCISE:
                #print("the action is exercise")
                #print("root_node asset price, ",root_node.state.asset_price)
                final_node = new_node.parent
                #print("final node asset price",final_node.state.asset_price)
                intrinsic_value = self.mdp.get_intrinsic_value(final_node.state.asset_price)
                #if self.verbose:
                #    print("the final reward is", intrinsic_value)
                return intrinsic_value
                #print("the action is hold")
            if new_node.state.time_step == self.mdp.T:
                intrinsic_value = self.mdp.get_intrinsic_value(new_node.state.asset_price)
                if self.verbose:
                    print("the final reward is", intrinsic_value)
                return intrinsic_value
            root_node = new_node
            #print("root_node asset price", root_node.state.asset_price)
            #print("price of root_node", root_node.state.asset_price)
            #self.detach_parent(root_node)

    def get_root_rewards(self):
        root_node = self.create_node(None, None, self.mdp.initial_state())
        root_node,action_taken, root_rewards = self.run_iteration(root_node, 200)
        return root_rewards
    
    def get_root_rewards_hindsight(self):
        root_node = self.create_node(None, None, self.mdp.initial_state())
        root_node,action_taken, root_rewards = self.run_iteration_hindsight(root_node, 200)
        return root_rewards

    def run_option_hindsight(self):
        root_node = self.create_node(None, None, self.mdp.initial_state())
        root_node_copy = self.create_node(None, None, self.mdp.initial_state())
        while True:
            new_node, action_taken,root_reward = self.run_iteration_hindsight(root_node, 300)
            #new_state = self.mdp.transition(root_node_copy.state, action_taken)
            if action_taken == USoptionAction.EXERCISE:
                final_node = new_node.parent
                #print("the parent node of transitioned node", final_node.state.asset_price)
                intrinsic_value = self.mdp.get_intrinsic_value(final_node.state.asset_price)
                if self.verbose:
                    print("the final reward is", intrinsic_value)
                return intrinsic_value
            if new_node.state.time_step == self.mdp.T:
                intrinsic_value = self.mdp.get_intrinsic_value(new_node.state.asset_price)
                if self.verbose:
                    print("the final reward is", intrinsic_value)
                return intrinsic_value
            root_node = new_node
            self.detach_parent(root_node)

    def run_iteration(self, node: StateNode[TState, TAction],iterations:int):
        root_rewards = [] 
        for i in range(iterations):
            #print("第几次",i)
            explore_node = self.select(node)
            expand_node = self.expand(explore_node)
            reward = self.simulate(expand_node)
            self.backpropagate(expand_node, reward)
            if self.verbose:
                print("root node reward",node.reward)
            root_rewards.append(node.reward)
        next_node,next_action = self.next(node)
        #print("next action is",next_action)
        return next_node,next_action, root_rewards
    
    def run_iteration_hindsight(self, node: StateNode[TState, TAction],iterations:int):
        root_rewards = [] 
        for i in range(iterations):
            if self.verbose:
                print("第几次",i)
            explore_node = self.select(node)
            expand_node = self.expand(explore_node)
            reward = self.simulate_hindsight(expand_node)
            self.backpropagate(expand_node, reward)
            root_rewards.append(node.reward)
        next_node,next_action = self.next(node)
        if self.verbose:
            print("next_action",next_action)
        return next_node,next_action, root_rewards

    def next(self,node: MENTSNode[TState, TAction]):

        if self.mdp.is_terminal(node.state):
            raise ValueError("Option has ended")

        action_probs_dict = self.q_estimator.get_softmax_prob_multinom(node.state, node.valid_actions)

        # Select the action with the highest probability
        best_action = max(action_probs_dict, key=action_probs_dict.get)
        #print("best action", best_action)
        best_child = None
        for child in node.children:
            if child.inducing_action == best_action:
                best_child = child
                break

        if best_child is None:
            raise RuntimeError("No child node found for the best action")

        return best_child, best_child.inducing_action
        #max_n = max(node.n for node in children)
        #print("number",max_n)

        #best_children = [c for c in children if c.n == max_n]
        #best_child = random.choice(best_children)
        #print("best child", best_child)

    def select(self, node: StateNode[TState, TAction]) -> StateNode[TState, TAction]:
        #print("selection start")
        current_node = node
        while True:
            current_node.valid_actions = self.mdp.actions(current_node.state)
            # If the node is terminal, return it
            if self.mdp.is_terminal(current_node.state):
                return current_node

            explored_actions = current_node.explored_actions()
            # This state has not been fully explored, return it
            if len(current_node.valid_actions) > len(explored_actions):
                #print("Node is not yet fully explored.")
                return current_node

            # This state has been explored, select best action
            #print("Node is fully explored.")
            action_probs_dict = self.q_estimator.get_softmax_prob_multinom(node.state, current_node.valid_actions)
            #print("Select action prbs dict",action_probs_dict)
            _, action_index = self.q_estimator.draw_from_multinomial(action_probs_dict)
            #print("action index", action_index)
            current_node = current_node.children[action_index]
            pass
            

    def expand(self, node: StateNode[TState, TAction], iteration_number = None) -> StateNode[TState, TAction]:
        if self.verbose:
            print("expand starts")
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node

        explored_actions = node.explored_actions()
        unexplored_actions = [a for a in node.valid_actions if a not in explored_actions]

        if len(unexplored_actions) == 0:
            raise RuntimeError("No unexplored actions available")

        action_taken = random.choice(unexplored_actions)

        new_state = self.mdp.transition(node.state, action_taken)
        return self.create_node(node, action_taken, new_state, node.n)
    
    def simulate_(self, node: StateNode[TState, TAction]) -> float:
        valid_actions = node.valid_actions
        if node.inducing_action == USoptionAction.EXERCISE:
            node = node.parent
            intrinsic_value = self.mdp.get_intrinsic_value(node.state.asset_price)
            return intrinsic_value
        else:
            current_state = node.state
            while True:
                random_action = random.choice(valid_actions)
                new_state = self.mdp.transition(current_state, random_action)
                if random_action == USoptionAction.EXERCISE:
                    intrinsic_value = self.mdp.get_intrinsic_value(current_state.asset_price)
                    return intrinsic_value
                if new_state.time_step == self.mdp.T or new_state.is_terminal == True:
                    intrinsic_value = self.mdp.get_intrinsic_value(new_state.asset_price)
                    return intrinsic_value
                current_state = new_state

    def simulate_hindsight(self, node: StateNode[TState, TAction]) -> float:
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
        if self.verbose:
            print("max_payff",max_payoff)
        return max_payoff

    
    def simulate(self, node: StateNode[TState, TAction], mc_sim_iter = 10) -> float:
        if self.verbose:
            print("Simulation:")

        if node.is_terminal:
            if self.verbose:
                print("Terminal state reached")
            parent = node.get_parent()
            parent_state = parent.state if parent != None else None
            return self.mdp.reward(parent_state, node.inducing_action, node.state)
        
        use_value_approx = np.random.uniform(0, 1) > self.alpha_value
        if self.value_function_estimator_callback is None:
            sim_reward, trajectory_history = self.simulate_by_simulation(node, mc_sim_iter = mc_sim_iter)
        else:
            # value_function_estimator_callback() will receive a StateNode object.
            if use_value_approx:
                sim_reward = self.value_function_estimator_callback(node)
            else:
                sim_reward, trajectory_history = self.simulate_by_simulation(node, mc_sim_iter = mc_sim_iter)
        
        if self.value_clipping and self.value_function_lower_estimator_callback is not None:
            sim_reward = np.max([sim_reward, self.value_function_lower_estimator_callback(node)])
        elif self.value_clipping and self.value_function_upper_estimator_callback is not None:
            if any(param for param in inspect.signature(self.value_function_upper_estimator_callback).parameters.values() if param.name == 'trajectory_history'):
                sim_reward = np.min([sim_reward, self.value_function_upper_estimator_callback(node, trajectory_history = trajectory_history)])
            else:
                sim_reward = np.min([sim_reward, self.value_function_upper_estimator_callback(node)])
        
        return sim_reward    
    
    def simulate_by_simulation(self, node, mc_sim_iter = 10):
        depth = 0
        current_state = node.state
        discount = self.discount_factor
        
        trajectory_history = []
        reward_history = []
        for i in range(mc_sim_iter):
            state_history = [current_state]
            while True:
                valid_actions = self.mdp.actions(current_state)
                random_action = random.choice(valid_actions)
                new_state = self.mdp.transition(current_state, random_action)
                state_history.append(new_state)

                if self.mdp.is_terminal(new_state):
                    reward = self.mdp.reward(current_state, random_action, new_state) * discount
                    if self.verbose:
                        print(f"-> Terminal state reached: {reward}")
                    reward_history.append(reward)
                    trajectory_history.append(state_history)
                    break

                current_state = new_state
                depth += 1
                discount *= self.discount_factor

                # statefulsolver, state should have a terminal check, in the state itself (ie last port in the schedule)
                if depth > self.simulation_depth_limit:
                    reward = self.mdp.reward(current_state, random_action, new_state) * discount
                    if self.verbose:
                        print(f"-> Depth limit reached: {reward}")
                    reward_history.append(reward)
                    trajectory_history.append(state_history)
                    break
        expected_reward = np.mean(reward_history)
        return expected_reward, trajectory_history

    def backpropagate(self, node: StateNode[TState, TAction], reward: float) -> None:
        current_state_node = node
        current_reward = reward

        while current_state_node != None:
            current_state_node.max_reward = max(current_reward, current_state_node.max_reward)
            
            current_state_node.reward += current_reward

            state_children_iter = current_state_node.children
            current_state_node.n += 1

            ### Check this over
            # get all taken actions to update q table
            all_poss_actions = [sc.inducing_action for sc in state_children_iter]
            action_visit_dict = {}
            
            for visited_action in all_poss_actions:
                for state_child in state_children_iter:
                    if state_child.inducing_action == visited_action:
                        if repr(visited_action) not in action_visit_dict:
                            action_visit_dict[repr(visited_action)] = 1
                        else:
                            action_visit_dict[repr(visited_action)] += 1
            
            reward_to_go_dict = {}
            for visited_action in all_poss_actions:
                for state_child in state_children_iter:
                    if state_child.inducing_action == visited_action:
                        reward_to_go_term = state_child.n / action_visit_dict[repr(state_child.inducing_action)] * self.q_estimator.get_state_value(state_child.state)
                        if repr(visited_action) not in reward_to_go_dict:
                            reward_to_go_dict[repr(visited_action)] = {"reward_to_go": reward_to_go_term, "action_obj": visited_action}
                        else:
                            reward_to_go_dict[repr(visited_action)]["reward_to_go"] += reward_to_go_term
            
            # Q and Value Update: ### Check this over...
            for action_repr in reward_to_go_dict.keys():
                self.q_estimator.update_q_value(current_state_node.state, reward_to_go_dict[action_repr]["action_obj"], reward, reward_to_go_dict[action_repr]["reward_to_go"], discount_factor=self.discount_factor)
            self.q_estimator.update_state_value(current_state_node.state, all_poss_actions)
            current_state_node.ments_value = self.q_estimator.get_state_value(current_state_node.state)

            if current_state_node.parent is None:
                self.ments_value_tracker.append(current_state_node.ments_value)

            current_state_node = current_state_node.parent
            current_reward *= self.discount_factor

    def create_node(self, parent: Optional[StateNode[TState, TAction]], inducing_action: Optional[TAction],
                    state: TState, number_of_visits=0) -> StateNode[TState, TAction]:
        
        valid_actions = self.mdp.actions(state)
        is_terminal = self.mdp.is_terminal(state)
        state_node = StateNode(parent, inducing_action, state, valid_actions, is_terminal)

        if parent != None:
            parent.add_child(state_node)

        return state_node