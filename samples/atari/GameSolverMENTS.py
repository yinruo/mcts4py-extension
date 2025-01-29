import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
import gymnasium as gym
import matplotlib.pyplot as plt
class GameSolverMENTS(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 exploration_constant: float,
                 simulation_depth_limit:float,
                 discount_factor: float,
                 temperature, 
                 epsilon,
                 env_name: str,
                 verbose: bool = False):
        self.mdp = mdp
        self.discount_factor = discount_factor
        self.simulation_depth_limit = simulation_depth_limit
        self.temperature = temperature
        self.epsilon = epsilon
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.env.reset()

        super().__init__(exploration_constant, verbose)

    def root(self) -> MENTSNode[TState, TAction]:
        self.__root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(self.__root_node)
        return self.__root_node
    
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
    
    def select(self, node: MENTSNode[TState, TAction]) -> MENTSNode[TState, TAction]:
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
            current_node = next((child for child in current_children if child.inducing_action == action), None)

    def expand(self, node: MENTSNode[TState, TAction], iteration_number=None) -> MENTSNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node

        current_children = node.children
        explored_actions = set([c.inducing_action for c in current_children])
        valid_action: set[TAction] = set(node.valid_actions)
        unexplored_actions = valid_action - explored_actions

        # Expand an unexplored action
        action_taken = random.sample(list(unexplored_actions), 1)[0]

        new_node = MENTSNode(node, action_taken)
        node.add_child(new_node)
        self.simulate_action(new_node)
        self.env.unwrapped.restore_state(node.state.current_state)
        observation, r_reward, terminated, truncated, _ = self.env.step(action_taken.value)
        node.action_reward[action_taken.value] = r_reward
        return new_node, action_taken


    def simulate(self, node: MENTSNode[TState, TAction], depth=0) -> float:
        self.env.unwrapped.restore_state(node.state.current_state)
        valid_actions = self.mdp.actions()
        total_reward = 0
        done = False
        while not done:
            random_action = random.choice(valid_actions)
            observation, reward, terminated, truncated, _ = self.env.step(random_action.value)
            done = terminated or truncated
            depth += 1
            total_reward += reward
            if done or self.simulation_depth_limit<depth:
                self.env.unwrapped.restore_state(node.state.current_state)
                break
        return total_reward 

    def backpropagate(self, node: MENTSNode[TState, TAction],action, reward: float) -> None:
        node.visits[action.value] += 1
        node.Q_sft[action.value] = node.action_reward[action.value] + reward
        softmax_value = self.softmax_value(node.Q_sft)
        inducing_action = node.inducing_action 
        node = node.parent 

        while node:
            node.visits[inducing_action.value] += 1
            node.Q_sft[inducing_action.value] = node.action_reward[inducing_action.value] + softmax_value
            
            if self.verbose:
                print("softmax value:", softmax_value)
                print("Q_sft:", node.Q_sft)

            softmax_value = self.softmax_value(node.Q_sft)
            inducing_action = node.inducing_action  
            node = node.parent 

    # Utilities

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
        for action in node.valid_actions:
            node.Q_sft[action.value] = 0.0

    def run_game(self):
        reward_episode = 0
        done = False
        root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(root_node)
        self.env = gym.make(self.env_name)
        self.env.reset()

        while not done : 
            root_node, action = self.run_game_iteration(root_node, 50)
            #print(action.value)
            _, reward, terminated, truncated, _ = self.env.step(action.value)
            reward_episode += reward
            done = terminated or truncated
            if done:
                print('MENTS episode reward: ',reward_episode)
                self.env.close()
                return reward_episode
  

    def run_game_iteration(self, node: MENTSNode[TState, TAction],iterations:int):
        for i in range(iterations):
            explore_node = self.select(node)
            expanded,action_taken = self.expand(explore_node)
            simulated_reward = self.simulate(expanded)
            self.backpropagate(expanded,action_taken, simulated_reward)
        next_node,next_action = self.next(node)
        #best_action_value = max(node.visits, key=lambda action_value: node.visits[action_value])

        #best_action = None
        #for action in node.valid_actions:
        #    if action.value == best_action_value:
        #        best_action = action
        #        break
        
        #best_child = None
        #for child in node.children:
        #    if child.inducing_action.value == best_action_value:
        #        best_child = child
        #        break

        return next_node, next_action
    
    def next(self,node: MENTSNode[TState, TAction]):

        if self.mdp.is_terminal(node.state):
            raise ValueError("Option has ended")
        
        soft_indmax_probs = self.soft_indmax(node.Q_sft)
        #print(soft_indmax_probs)
        index_of_better_value = np.argmax(soft_indmax_probs)
        #print(index_of_better_value)

        best_child = None
        for child in node.children:
            if child.inducing_action.value == index_of_better_value:
                best_child = child
                break

        #children = node.children
        #max_n = max(node.n for node in children)
        #best_children = [c for c in children if c.n == max_n]
        #best_child = random.choice(best_children)

        return best_child, best_child.inducing_action
    

    def next_test(self,node: MENTSNode[TState, TAction]):

        children = node.children
        max_n = max(node.n for node in children)
        # print(max_n)
        best_children = [c for c in children if c.n == max_n]
        best_child = random.choice(best_children)

        return best_child, best_child.inducing_action





