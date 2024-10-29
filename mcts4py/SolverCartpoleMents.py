import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
import gymnasium as gym
from copy import deepcopy
import time

class SolverCartpoleMents(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 exploration_constant: float,
                 discount_factor: float,
                 temperature, 
                 epsilon,
                 env_name: str,
                 verbose: bool = False):

        self.mdp = mdp
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.epsilon = epsilon
        self.env_name = env_name
        #self.__root_node = MENTSNode[TState, TAction](None, None)
        #self.simulate_action(self.__root_node)

        super().__init__(exploration_constant, verbose)

    def root(self) -> MENTSNode[TState, TAction]:
        self.__root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(self.__root_node)
        self.reset_node(self.__root_node)
        return self.__root_node
    
    def e2w(self, node: MENTSNode[TState, TAction]):
        total_visits = 0
        for action in node.valid_actions:
            total_visits += node.n[action.value]
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
    
    def reset_node(self, node: MENTSNode[TState, TAction]):
        # Reset the node's statistics
        node.n = defaultdict(float)  
        for action in node.valid_actions:
            node.n[action.value] = 0.0
        node.max_reward = float('-inf')
        node.depth = 0.0
        node.Q_sft = defaultdict(float)  
        node.reward = defaultdict(float)
        for action in node.valid_actions:
            node.Q_sft[action.value] = 0.0

    def select(self, node: MENTSNode[TState, TAction], iteration_number=None) -> MENTSNode[TState, TAction]:
        current_node = node
        while True:
            current_children = current_node.children
            explored_actions = set([c.inducing_action for c in current_children])
            if len(set(current_node.valid_actions) - explored_actions) > 0:
                return current_node
            action = self.e2w(current_node)
            current_node = next((child for child in current_children if child.inducing_action == action), None)
        
    def expand(self, node: MENTSNode[TState, TAction], iteration_number=None) -> MENTSNode[TState, TAction]:
        current_children = node.children
        valid_action: set[TAction] = set(node.valid_actions)
        explored_actions = set([c.inducing_action for c in current_children])
        unexplored_actions = valid_action - explored_actions
        action_taken = random.sample(list(unexplored_actions), 1)[0]
        new_node = MENTSNode(node, action_taken)
        node.add_child(new_node)
        self.simulate_action(new_node)
        self.reset_node(new_node)
        new_game = deepcopy(node.state.env)
        observation, r_reward, terminated, truncated, _ = new_game.step(action_taken.value)
        node.reward[action_taken.value] = r_reward
        return new_node, action_taken

    def simulate(self, node: MENTSNode[TState, TAction]) -> float:
        new_game = deepcopy(node.state.env)
        valid_actions = self.mdp.actions(node.state)
        total_reward = 0
        done = False
        while not done:
            random_action = random.choice(valid_actions)
            observation, reward, terminated, truncated, _ = new_game.step(random_action.value)
            done = terminated or truncated
            total_reward += reward
            if done:
                new_game.close()
                break
        return total_reward
                

    def backpropagate(self, node: MENTSNode[TState, TAction], action, reward: float) -> None:
        # Step 1: 更新叶节点的父节点
        node.n[action.value] += 1
        node.Q_sft[action.value] = node.reward[action.value] + reward
        softmax_value = self.softmax_value(node.Q_sft)
        inducing_action = node.inducing_action  # 获取叶节点的父节点所带来的 action
        node = node.parent  # 向上移动一个节点

        # Step 2: 对于非叶节点的递归更新，使用 softmax backup
        while node:
            # 更新当前节点的访问次数和 Q 值
            node.n[inducing_action.value] += 1
            node.Q_sft[inducing_action.value] = node.reward[inducing_action.value] + softmax_value
            
            if self.verbose:
                print("softmax value:", softmax_value)
                print("Q_sft:", node.Q_sft)

            # 计算当前节点的 Q_sft 的 softmax 值，并用于更新上层节点
            softmax_value = self.softmax_value(node.Q_sft)
            inducing_action = node.inducing_action  # 更新为当前节点的 inducing_action
            node = node.parent  # 向上移动到上一级节点



    def simulate_action(self, node: MENTSNode[TState, TAction]):
        # If this is a top node, initialize the parameter for the node 
        if node.parent == None:
            initial_state = self.mdp.initial_state()
            node.state = initial_state
            node.valid_actions = self.mdp.actions(initial_state)
            return

        if node.inducing_action == None:
            raise RuntimeError("Action was null for non-null parent")
        new_state = self.mdp.transition(node.parent.state, node.inducing_action)
        node.state = new_state
        node.valid_actions = self.mdp.actions(node.state) 

    def detach_parent(self,node: MENTSNode[TState, TAction]):
        del node.parent
        node.parent = None

    def next(self,node: MENTSNode[TState, TAction]):

        if self.mdp.is_terminal(node.state):
            raise ValueError("game has ended")

        children = node.children
        max_n = max(node.n for node in children)

        best_children = [c for c in children if c.n == max_n]
        best_child = random.choice(best_children)

        return best_child, best_child.inducing_action


    def reset_root_node(self):
        self.__root_node = MENTSNode[TState, TAction](None, None)
        self.simulate_action(self.__root_node)

    def run_game(self, episodes: int):
        rewards = []
        moving_average = []
        for e in range(episodes):
            reward_episode = 0
            done = False
            root_node = self.root()
            game = gym.make(self.env_name)
            game.reset()
            root_node.state.env = deepcopy(game.unwrapped)
            print('episode #' + str(e+1))

            while not done:
                root_node, action = self.run_game_iteration(root_node,50)
                observation, reward, terminated, truncated, _ = game.step(action.value)
                reward_episode += reward
                done = terminated or truncated

                if done:
                    print('reward ' + str(reward_episode))
                    game.close()
                    break
            
            rewards.append(reward)
            moving_average.append(np.mean(rewards[-100:]))
    def run_game_iteration(self, node: MENTSNode[TState, TAction],iterations:int):
        for i in range(iterations):
            current_node = self.select(node)
            expand_new_node, action_taken = self.expand(current_node)
            reward = self.simulate(expand_new_node)
            self.backpropagate(expand_new_node,action_taken, reward)
        #next_node, next_action = self.next(node)
        best_action_value = max(node.n, key=lambda action_value: node.n[action_value])

        best_action = None
        for action in node.valid_actions:
            if action.value == best_action_value:
                best_action = action
                break
        
        best_child = None
        for child in node.children:
            if child.inducing_action.value == best_action_value:
                best_child = child
                break
        
        return best_child, best_action
    
    def run_random_game(self, episodes: int):
        rewards = []
        moving_average = []
        for e in range(episodes):
            reward_episode = 0
            done = False
            root_node = self.root()
            game = gym.make(self.env_name)
            game.reset()
            print('episode #' + str(e+1))

            while not done:
                action = game.action_space.sample()
                observation, reward, terminated, truncated, _ = game.step(action)
                reward_episode += reward
                done = terminated or truncated

                if done:
                    print('reward ' + str(reward_episode))
                    game.reset()
                    game.close()
                    break