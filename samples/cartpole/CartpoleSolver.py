import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
import gymnasium as gym
from copy import deepcopy
import time

class CartpoleSolver(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 exploration_constant: float,
                 discount_factor: float,
                 env_name: str,
                 iteration_time : int,
                 verbose: bool = False):

        self.mdp = mdp
        self.discount_factor = discount_factor
        self.env_name = env_name
        self.iteration_time = iteration_time
        #self.__root_node = ActionNode[TState, TAction](None, None)
        #self.simulate_action(self.__root_node)

        super().__init__(exploration_constant, verbose)

    def root(self) -> ActionNode[TState, TAction]:
        self.__root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(self.__root_node)
        return self.__root_node

    def select(self, node: ActionNode[TState, TAction], iteration_number=None) -> ActionNode[TState, TAction]:
        current_node = node
        while True:
            current_children = current_node.children
            explored_actions = set([c.inducing_action for c in current_children])
            if len(set(current_node.valid_actions) - explored_actions) > 0:
                return current_node
            max_uct_value = max(self.calculate_uct(c) for c in current_children)
            best_children = [c for c in current_children if self.calculate_uct(c) == max_uct_value]
            current_node = random.choice(best_children)
        
        

    def expand(self, node: ActionNode[TState, TAction], iteration_number=None) -> ActionNode[TState, TAction]:
        current_children = node.children
        valid_action: set[TAction] = set(node.valid_actions)
        explored_actions = set([c.inducing_action for c in current_children])
        unexplored_actions = valid_action - explored_actions
        action_taken = random.sample(list(unexplored_actions), 1)[0]
        new_node = ActionNode(node, action_taken)
        node.add_child(new_node)
        self.simulate_action(new_node)
        return new_node


    def simulate(self, node: ActionNode[TState, TAction]) -> float:
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
                break
        return total_reward
                

    def backpropagate(self, node: ActionNode[TState, TAction], reward: float) -> None:
        while node is not None:
            node.reward += reward
            node.n += 1
            node = node.parent


    def simulate_action(self, node: ActionNode[TState, TAction]):
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

    def detach_parent(self,node: ActionNode[TState, TAction]):
        del node.parent
        node.parent = None


    def reset_root_node(self):
        self.__root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(self.__root_node)

    def run_game(self, episodes: int):
        rewards = []
        for e in range(episodes):
            reward_episode = 0
            done = False
            root_node = self.root()
            game = gym.make(self.env_name)
            game.reset()
            root_node.state.env = deepcopy(game.unwrapped)
            print('episode #' + str(e+1))

            while not done:
                root_node, action = self.run_game_iteration(root_node, self.iteration_time)
                observation, reward, terminated, truncated, _ = game.step(action.value)
                reward_episode += reward
                done = terminated or truncated

                if done:
                    print('reward ' + str(reward_episode))
                    rewards.append(reward_episode)
                    game.close()
                    break
            
            
        average_reward = sum(rewards) / len(rewards)
        print('Average Reward: ' + str(average_reward))

        return average_reward

    def run_game_iteration(self, node: ActionNode[TState, TAction],iterations:int):
        for i in range(iterations):
            current_node = self.select(node)
            expand_new_node = self.expand(current_node)
            reward = self.simulate(expand_new_node)
            self.backpropagate(expand_new_node, reward)
        next_node, next_action = self.next(node)
        return next_node, next_action
    
    def next(self,node: ActionNode[TState, TAction]):

        if self.mdp.is_terminal(node.state):
            raise ValueError("game has ended")

        children = node.children
        max_n = max(node.n for node in children)

        best_children = [c for c in children if c.n == max_n]
        best_child = random.choice(best_children)

        return best_child, best_child.inducing_action
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
                    game.close()
                    break