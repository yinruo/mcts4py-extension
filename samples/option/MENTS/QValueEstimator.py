import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

LAMBDA_MENTS_DECAY_RATE = 10.5
ACTION_SPACE_SIZE = 200
DISCOUNT_FACTOR = 0.9
MENTS_ALPHA = 0.5
class StateValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StateValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

class StateActionValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StateActionValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state_action):
        x = torch.relu(self.fc1(state_action))
        return self.fc2(x)


def exponential_decay(num_calls, decay_rate=LAMBDA_MENTS_DECAY_RATE):
    initial_value = 1.0  # Initial value
    result = initial_value * math.exp(-decay_rate * num_calls)
    return result
class QValueEstimator:
    def __init__(self, alpha=MENTS_ALPHA, lambda_temp_callback=exponential_decay, action_space_size=ACTION_SPACE_SIZE):
        # self.state_value_net = StateValueNetwork(state_dim, hidden_dim, 1)
        # self.q_net = StateActionValueNetwork(state_dim + action_dim, hidden_dim, 1)
        # self.optimizer_state_value = optim.Adam(self.state_value_net.parameters(), lr=0.001)
        # self.optimizer_q_value = optim.Adam(self.q_net.parameters(), lr=0.001)
        # self.loss_fn = nn.MSELoss()
        self.q_val_dict = {}
        self.state_val_dict = {}
        self.alpha = alpha
        self.num_calls = 0
        self.lambda_temp_callback = lambda_temp_callback
        self.action_space_size = action_space_size
    
    # def predict_state_value(self, state):
    #     state = torch.tensor(state, dtype=torch.float32)
    #     self.num_calls += 1
    #     return self.state_value_net(state)
    def get_state_value(self, state):
        if repr(str(state)) in self.state_val_dict:
            state_value = self.state_val_dict[repr(str(state))]
        else:
            self.state_val_dict[repr(str(state))] = np.random.uniform()
            state_value = self.state_val_dict[repr(str(state))]
        return state_value
    
    def get_q_value(self, state, action):
        # state_action = torch.tensor(np.concatenate((state, action)), dtype=torch.float32)
        # return self.q_net(state_action)
        if repr([str(state), str(action)]) in self.q_val_dict:
            q_value = self.q_val_dict[repr([str(state), str(action)])]
        else:
            self.q_val_dict[repr([str(state), str(action)])] = np.random.uniform()
            q_value = self.q_val_dict[repr([str(state), str(action)])]
        return q_value
    
    def get_max_q_value(self, state, possible_actions):
        max_q_value = -np.inf
        for a in possible_actions:
            candidate_q_value = self.get_q_value(state, a)
            if candidate_q_value > max_q_value:
                max_q_value = candidate_q_value

        return max_q_value
    
    def update_state_value(self, state, possible_actions):
        # Handle empty possible_actions
        if not possible_actions:
            self.state_val_dict[repr(str(state))] = float('-inf')  # Assign a very low value for empty actions
            return float('-inf')

        # Compute q_values for the given possible actions
        q_values = [self.get_q_value(state, a) for a in possible_actions]
        #print("q_values", q_values)

        # Stabilize using the log-sum-exp trick
        max_q_value = max(q_values)  # Ensure q_values is not empty
        val_term = sum(np.exp((1/self.alpha) * (q_value - max_q_value)) for q_value in q_values)
        new_value = self.alpha * (np.log(val_term) + (1/self.alpha) * max_q_value)

        # Update state value dictionary
        self.state_val_dict[repr(str(state))] = new_value
        return new_value

    
    def update_q_value(self, state, action, reward, reward_to_go, discount_factor=DISCOUNT_FACTOR):
        if np.isinf(reward):
            reward = 0
        if np.isinf(reward_to_go):
            reward_to_go = 0
        self.q_val_dict[repr([str(state), str(action)])] = reward + discount_factor*reward_to_go
    
    def get_softmax_prob_per_action(self, state, action, lambda_temp_term=None):
        q_val = self.get_q_value(state, action)
        state_value = self.get_state_value(state)
        rho_ments = np.exp((1/self.alpha) * (q_val - state_value ))
        if lambda_temp_term is None:
            decay_func = self.lambda_temp_callback
        else:
            decay_func = lambda x: lambda_temp_term
        decay_term = decay_func(self.num_calls)
        prob_ments = (1 - decay_term)*rho_ments + decay_term/self.action_space_size
        return prob_ments
    
    def get_softmax_prob_multinom(self, state, possible_actions):
        action_probs_dic = {}
        for a in possible_actions:
            action_probs_dic[a] = self.get_softmax_prob_per_action(state, a)
        return action_probs_dic

    def draw_from_multinomial(self, action_prob_dict):
        # Extract actions and unnormalized probabilities
        actions = list(action_prob_dict.keys())
        unnormalized_probs = list(action_prob_dict.values())
        normalized_probs = np.array(unnormalized_probs) / np.sum(unnormalized_probs)
        if np.isnan(normalized_probs).any():
            # If NaNs exist, replace them with uniform probabilities
            normalized_probs[np.isnan(normalized_probs)] = 1.0 / len(normalized_probs)
        action_index = np.random.multinomial(1, normalized_probs).argmax()
        # Return the action corresponding to the drawn index
        self.num_calls += 1
        return actions[action_index], action_index
    
    # def forward_bellman_update(self, state, action, reward, next_state, done, discount_factor):
    #     next_state_value = self.predict_state_value(next_state).item()
    #     target = reward + discount_factor * next_state_value * (1 - done)
    #     self.update_state_action_value(state, action, target)

# self.optimizer_state_value.zero_grad()
# loss = self.loss_fn(max_q_value, max_q_value)
# loss.backward()
# self.optimizer_state_value.step()

# def update_q_value(self, state, action, target):
#     state_action = torch.tensor(np.concatenate((state, action)), dtype=torch.float32)
#     target = torch.tensor(target, dtype=torch.float32)
#     self.optimizer_q_value.zero_grad()
#     predicted = self.q_net(state_action)
#     loss = self.loss_fn(predicted, target)
#     loss.backward()
#     self.optimizer_q_value.step()

# def softmax_policy_update(self, state, action_values, temperature=1.0):
#     action_probs = nn.functional.softmax(action_values / temperature, dim=0)
#     chosen_action = torch.multinomial(action_probs, num_samples=1).item()
#     chosen_action_prob = action_probs[chosen_action]
#     target = self.predict_state_value(state).detach().item()  # Detach to prevent gradients from flowing
#     error = target - chosen_action_prob
#     self.update_state_action_value(state, chosen_action, error)