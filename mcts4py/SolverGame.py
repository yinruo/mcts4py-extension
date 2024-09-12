import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
import gymnasium as gym
import matplotlib.pyplot as plt
class SolverGame(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 discount_factor: float,
                 env_name: str,
                 verbose: bool = False):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        self.env_name = env_name
        self.__root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(self.__root_node)
        self.env = gym.make(self.env_name)
        self.env.reset()

        super().__init__(exploration_constant, verbose)

    def root(self) -> ActionNode[TState, TAction]:
        return self.__root_node

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

    # def simulate(self, node: ActionNode[TState, TAction]) -> float:
    #     if self.verbose:
    #         print("Simulation:")
    #
    #     if self.mdp.is_terminal(node.state):
    #         if self.verbose:
    #             print("Terminal state reached")
    #         parent = node.get_parent()
    #         parent_state = parent.state if parent != None else None
    #         return self.mdp.reward(parent_state, node.inducing_action, node.state)
    #
    #     depth = 0
    #     current_state = node.state
    #     discount = self.discount_factor
    #
    #     while True:
    #         valid_actions = self.mdp.actions(current_state)
    #         random_action = random.choice(valid_actions)
    #         new_state = self.mdp.transition(current_state, random_action)
    #
    #         if self.mdp.is_terminal(new_state):
    #             reward = self.mdp.reward(current_state, random_action, new_state) * discount
    #             if self.verbose:
    #                 print(f"-> Terminal state reached: {reward}")
    #             return reward
    #
    #         current_state = new_state
    #         depth += 1
    #         discount *= self.discount_factor
    #
    #         if depth > self.simulation_depth_limit:
    #             reward = self.mdp.reward(current_state, random_action, new_state) * discount
    #             if self.verbose:
    #                 print(f"-> Depth limit reached: {reward}")
    #             return reward

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

    # Utilities

    def simulate_action(self, node: ActionNode[TState, TAction]):
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

    def reset_tree(self, node: ActionNode[TState, TAction]):
        # Reset the node's statistics
        node.n = 0
        self.reward = 0.0
        self.max_reward = 0.0
        node.__children = []

    def run_game(self, episodes: int):
        rewards = []
        total_reward = 0
        for e in range(episodes):
            reward_episode = 0
            done = False
            root_node = ActionNode[TState, TAction](None, None)
            self.simulate_action(root_node)
            self.reset_tree(root_node)
            initial_s = self.env.unwrapped.clone_state(include_rng=True)
            print('episode #' + str(e+1))

            while not done : 
                root_node, action = self.run_game_iteration(root_node, 10)
                print(action.value)
                observation, reward, terminated, truncated, _ = self.env.step(action.value)
                reward_episode += reward
                done = terminated or truncated

                if done:
                    print('reward ' + str(reward_episode))
                    rewards.append(reward_episode)
                    total_reward += reward_episode 
                    self.env.close()
                    break
        average_reward = total_reward / episodes if episodes > 0 else 0
        return rewards, average_reward

  

    def run_game_iteration(self, node: ActionNode[TState, TAction],iterations:int):
        for i in range(iterations):
            explore_node = self.select(node)
            expanded = self.expand(explore_node)
            simulated_reward = self.simulate(expanded)
            self.backpropagate(expanded, simulated_reward)
        next_node, next_action = self.next(node)
        return next_node, next_action
    

    def next(self,node: ActionNode[TState, TAction]):

        children = node.children
        max_n = max(node.n for node in children)
        # print(max_n)
        best_children = [c for c in children if c.n == max_n]
        best_child = random.choice(best_children)

        return best_child, best_child.inducing_action

    def run_random_game(self, episodes: int):
        rewards = []
        total_reward = 0
        for e in range(episodes):
            reward_episode = 0
            action_count = 0
            done = False
            root_node = self.root()
            game = gym.make(self.env_name)
            game.reset()
            print('episode #' + str(e+1))

            while not done:
                action = game.action_space.sample()
                observation, reward, terminated, truncated, _ = game.step(action)
                action_count += 1
                reward_episode += reward
                done = terminated or truncated

                if done:
                    print('reward ' + str(reward_episode))
                    rewards.append(reward_episode)
                    total_reward += reward_episode 
                    game.close()
                    break
        average_reward = total_reward / episodes if episodes > 0 else 0
        return rewards, average_reward
