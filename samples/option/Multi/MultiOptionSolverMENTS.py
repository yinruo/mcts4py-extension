import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
from samples.option.USoptionMDP import USoptionAction
from samples.option.Multi.USMultiOptionMDP import USMultiOptionMDP,MultiOptionAction
class MultiOptionSolverMENTS(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):
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
        self.root_node = MultiMENTSNode[TState, TAction](None, None)
        self.simulate_action(self.root_node)

        super().__init__(exploration_constant, verbose)   

    def root(self) -> MultiMENTSNode[TState, TAction]:
        return self.root_node 
    
    def e2w(self, node: MultiMENTSNode[TState, TAction]):
        total_visits = sum(node.visits[tuple(action.decisions)] for action in node.valid_actions)
        lambda_s = 1.0 if total_visits == 0 else (self.epsilon * len(node.valid_actions)) / np.log(total_visits + 1)

        soft_indmax_probs = self.soft_indmax(node.Q_sft)
        action_probabilities = []

        for action in node.valid_actions:
            key = tuple(action.decisions)
            uniform_prob = 1 / len(node.valid_actions)
            soft_prob = soft_indmax_probs[list(node.Q_sft.keys()).index(key)]
            combined_prob = (1 - lambda_s) * soft_prob + lambda_s * uniform_prob
            action_probabilities.append(combined_prob)

        action_probabilities = np.array(action_probabilities)
        action_probabilities = np.nan_to_num(action_probabilities, nan=0.0)
        action_probabilities[action_probabilities < 0] = 0.0

        total = action_probabilities.sum()
        if total == 0:
            action_probabilities = np.ones_like(action_probabilities)
            total = action_probabilities.sum()
        action_probabilities /= total

        return np.random.choice(node.valid_actions, p=action_probabilities)

    

    def softmax_value(self, Q_values: dict) -> float:
        keys = list(Q_values.keys())
        values = np.array([Q_values[k] for k in keys])
        max_Q = np.max(values)
        exp_values = np.exp((values - max_Q) / self.temperature)
        sum_exp = np.sum(exp_values)
        return self.temperature * np.log(sum_exp) + max_Q

    def soft_indmax(self, Q_values: dict) -> np.ndarray:
        softmax_val = self.softmax_value(Q_values)
        keys = list(Q_values.keys())
        values = np.array([Q_values[k] for k in keys])
        probs = np.exp((values - softmax_val) / self.temperature)
        return probs / np.sum(probs)

    def select(self, node: MultiMENTSNode[TState, TAction]) -> MultiMENTSNode[TState, TAction]:
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
            action = self.e2w(current_node)
            current_node = next(
                (child for child in current_children if tuple(child.inducing_action.decisions) == tuple(action.decisions)), None
            )

    def expand(self, node: MultiMENTSNode[TState, TAction], iteration_number=None) -> MultiMENTSNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node, node.inducing_action
        
        valid_actions = node.valid_actions
        random_action = random.choice(valid_actions)
        new_node = MultiMENTSNode(node, random_action)
        node.add_child(new_node)
        self.simulate_action(new_node)
        
        return new_node, new_node.inducing_action
    
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

    def backpropagate(self, node: MultiMENTSNode[TState, TAction], action: MultiOptionAction, reward: float):
        key = tuple(action.decisions)
        node.visits[key] += 1
        if key not in node.action_reward:
            node.action_reward[key] = 0.0
        node.Q_sft[key] = node.action_reward[key] + reward
        node.reward += reward
        softmax_val = self.softmax_value(node.Q_sft)
        parent = node.parent
        while parent:
            key = tuple(node.inducing_action.decisions)
            parent.visits[key] += 1
            parent.Q_sft[key] = parent.action_reward[key] + softmax_val
            parent.reward += reward
            reward *= self.discount_factor
            node = parent
            parent = parent.parent
    
    def simulate_action(self, node: MultiMENTSNode[TState, TAction]):
        if node.parent == None:
            initial_state = self.mdp.initial_state()
            node.state = initial_state
            node.valid_actions = self.mdp.actions(node.state)
            for action in node.valid_actions:
                key = tuple(action.decisions)
                node.Q_sft[key] = 0.0
                node.action_reward[key] = 0.0
            return

        if node.inducing_action == None:
            raise RuntimeError("Action was null for non-null parent")
        new_state = self.mdp.transition(node.parent.state, node.inducing_action)
        node.state = new_state
        node.valid_actions = self.mdp.actions(node.state)   
        for action in node.valid_actions:
            key = tuple(action.decisions)
            node.Q_sft[key] = 0.0
            node.action_reward[key] = 0.0
        
    def detach_parent(self,node: MultiMENTSNode[TState, TAction]):
        del node.parent
        node.parent = None
        
    def run_iteration(self, node: MultiMENTSNode[TState, TAction],iterations:int):
        root_rewards = []
        for i in range(iterations):
            explore_node = self.select(node)
            expanded_node,action_taken = self.expand(explore_node)
            simulated_reward = self.simulate(expanded_node)
            self.backpropagate(expanded_node,action_taken, simulated_reward)
            root_rewards.append(node.reward)
        next_node,next_action = self.next(node)
        return next_node,next_action,root_rewards
    
    def run_iteration_hindsight(self, node: MultiMENTSNode[TState, TAction],iterations:int):
        root_rewards = []
        for i in range(iterations):
            explore_node = self.select(node)
            expanded_node,action_taken = self.expand(explore_node)
            simulated_reward =self.simulate_hindsight(expanded_node)
            self.backpropagate(expanded_node,action_taken, simulated_reward)
            root_rewards.append(node.reward)
        next_node,next_action = self.next(node)
        return next_node,next_action,root_rewards


    def next(self, node: MultiMENTSNode[TState, TAction]):
        if self.mdp.is_terminal(node.state):
            raise ValueError("Option has ended")

        if not node.children:
            random_action = random.choice(node.valid_actions)
            fallback_child = MultiMENTSNode(node, random_action)
            node.add_child(fallback_child)
            self.simulate_action(fallback_child)
            return fallback_child, random_action

        soft_indmax_probs = self.soft_indmax(node.Q_sft)
        keys = list(node.Q_sft.keys())
        best_key = keys[np.argmax(soft_indmax_probs)]

        matching_action = next(
            (a for a in node.valid_actions if tuple(a.decisions) == best_key),
            None
        )
        if matching_action is None:
            random_action = random.choice(node.valid_actions)
            fallback_child = MultiMENTSNode(node, random_action)
            node.add_child(fallback_child)
            self.simulate_action(fallback_child)
            return fallback_child, random_action

        matching_child = next(
            (c for c in node.children if tuple(c.inducing_action.decisions) == tuple(matching_action.decisions)),
            None
        )

        if matching_child is None:
            matching_child = MultiMENTSNode(node, matching_action)
            node.add_child(matching_child)
            self.simulate_action(matching_child)

        return matching_child, matching_child.inducing_action

    
    def run_option(self):
        root_node = MultiMENTSNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        reward = 0.0

        while True:
            root_node, action, _ = self.run_iteration(root_node, 400)
            prev_state = root_node.parent.state
            #print("method 3, action is", action.decisions)

            prev_flags = root_node.parent.state.exercised_flags[:]
            #print("method 3, old flags, ",prev_flags)
            next_state = self.mdp.transition(prev_state, action)
            #next_state = root_node.state
            #root_node.state = next_state 

            new_flags = next_state.exercised_flags
            #print("method 3, new flags, ",new_flags)
            exercised_now = [
                i for i, (prev, new) in enumerate(zip(prev_flags, new_flags))
                if not prev and new
            ]
            #print("method 3, exercised_now", exercised_now)
            print("method 3, next state accet prices",next_state.asset_prices )
            for i in exercised_now:
                price = next_state.asset_prices[i]
                K = self.mdp.K_list[i]
                opt_type = self.mdp.option_type_list[i]
                val = self.mdp.get_intrinsic_value(price, K, opt_type)
                #print("method 3, val is added to the reward, val=", val)
                reward += val

            if all(next_state.exercised_flags):
                #print("method 3, final reward", reward)
                return reward

            if next_state.time_step == self.mdp.T:
                for i, was_exercised in enumerate(next_state.exercised_flags):
                    if not was_exercised:
                        price = next_state.asset_prices[i]
                        K = self.mdp.K_list[i]
                        opt_type = self.mdp.option_type_list[i]
                        val = self.mdp.get_intrinsic_value(price, K, opt_type)
                        if val > 0:
                            #print("method 3, val is added to the reward at terminal state, val=", val)
                            reward += val
                #print("method 3, final reward", reward)
                return reward
            root_node.state = next_state

            self.detach_parent(root_node)


    def run_option_hindsight(self):
        root_node = MultiMENTSNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        reward = 0.0

        while True:
            root_node, action, _ = self.run_iteration_hindsight(root_node, 400)

            prev_flags = root_node.parent.state.exercised_flags[:]
            prev_state = root_node.parent.state

            next_state = self.mdp.transition(prev_state, action)
            #next_state = root_node.state
            root_node.state = next_state 

            exercised_now = [
                i for i, (prev, new) in enumerate(zip(prev_flags, next_state.exercised_flags))
                if not prev and new
            ]

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

            self.detach_parent(root_node)
