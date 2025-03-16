from abc import ABC

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim

from config import (buffer_capacity, history_size, batch_size,
                    fc_sizes, lstm_state_size, learning_rate)

th.cuda.empty_cache()


class ReplayBuffer(ABC):

    def __init__(self, num_users, num_features, device):
        super().__init__()

        self.num_users = num_users
        self.num_features = num_features

        self.capacity = buffer_capacity
        self.history_size = history_size
        self.batch_size = batch_size
        self.device = device

        self.state_memory = np.zeros(
            (self.capacity, self.num_users, self.history_size, self.num_features),
            dtype=np.float32)
        self.next_state_memory = np.zeros(
            (self.capacity, self.num_users, self.history_size, self.num_features),
            dtype=np.float32)
        self.action_memory = np.zeros((self.capacity, self.num_users), dtype=np.int64)
        self.reward_memory = np.zeros((self.capacity, self.num_users), dtype=np.float32)
        self.terminal_memory = np.zeros((self.capacity, self.num_users), dtype=bool)

        self.mem_counter = 0

    def store_experience(self, state, next_state, action, reward, done):
        index = self.mem_counter % self.capacity

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample_buffer(self):
        max_mem = min(self.mem_counter, self.capacity)

        # "replace=False" assures that no repetitive memory is selected in batch
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = th.tensor(self.state_memory[batch]).to(self.device)
        next_state = th.tensor(self.next_state_memory[batch]).to(self.device)
        actions = th.tensor(self.action_memory[batch]).to(self.device)
        rewards = th.tensor(self.reward_memory[batch]).to(self.device)
        terminal = th.tensor(self.terminal_memory[batch]).to(self.device)

        experience = [states, next_state, actions, rewards, terminal]

        return experience

    def flush(self):
        ...

    def get_all_data(self):
        return self.state_memory, self.next_state_memory, \
            self.action_memory, self.reward_memory, self.terminal_memory, \
            self.mem_counter

    def set_all_data(self, state_memory, next_state_memory,
                     action_memory, reward_memory, terminal_memory,
                     mem_counter):
        self.state_memory = state_memory
        self.next_state_memory = next_state_memory
        self.action_memory = action_memory
        self.reward_memory = reward_memory
        self.terminal_memory = terminal_memory
        self.mem_counter = mem_counter


class DeepQNetwork(nn.Module):
    # Reference: https://github.com/mshokrnezhad/Dueling_for_DRL

    def __init__(self, name, num_features, num_actions, device):
        nn.Module.__init__(self)
        self.name = name
        self.num_features = num_features
        self.num_actions = num_actions
        self.device = device

        self.fc_sizes = fc_sizes
        self.lstm_state_size = lstm_state_size
        self.learning_rate = learning_rate

        # Build the Modules (LSTM + FC)
        self.lstm = nn.LSTM(self.num_features, self.lstm_state_size, batch_first=True)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.fc_1 = nn.Linear(self.lstm_state_size, self.fc_sizes[0])
        self.fc_2 = nn.Linear(self.fc_sizes[0], self.fc_sizes[1])

        self.V = nn.Linear(self.fc_sizes[1], 1)
        self.A = nn.Linear(self.fc_sizes[1], self.num_actions)

        # self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate,
                                    amsgrad=True, weight_decay=0.001)

        self.to(self.device)  # move whole model to the device

        # to avoid memory issues
        self.lstm.flatten_parameters()

    def forward(self, state):
        # forward propagation includes defining layers

        features, _ = self.lstm(state)
        x = self.leaky_relu(features[:, -1, :])

        x = self.leaky_relu(self.fc_1(x))
        x = self.leaky_relu(self.fc_2(x))

        V = self.V(x)
        A = self.A(x)

        return V, A

    def save_checkpoint(self, checkpoint_file):
        th.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(th.load(checkpoint_file))
