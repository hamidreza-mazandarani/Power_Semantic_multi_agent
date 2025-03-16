from copy import deepcopy

import numpy as np
import torch as th
import torch.cuda
import torch.nn as nn
import tqdm

from config import (load_pretrained_model, batch_size, replace_target_interval, gamma,
                    num_episodes_train, num_time_slots_default,
                    epsilon_init, epsilon_decay, epsilon_min,
                    verbose_default, save_model_after_train)
from model import ReplayBuffer, DeepQNetwork


def get_numpy_from_dict_values(x):
    return np.array(list(x.values()))


class MA_D3QL:
    def __init__(self, num_users, num_channels, power_level_all_channels, num_features,
                 algorithm, model_name='ma_d3ql'):

        self.num_users = num_users
        self.num_channels = num_channels
        self.power_level_all_channels = power_level_all_channels
        self.algorithm = algorithm
        self.model_name = model_name
        self.num_features = num_features

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # read from config
        self.load_pretrained_model = load_pretrained_model
        self.batch_size = batch_size
        self.replace_target_interval = replace_target_interval
        self.gamma = gamma

        self.buffer = ReplayBuffer(self.num_users, self.num_features, self.device)

        self.loss = nn.MSELoss()

        self.epsilon = epsilon_init

        # create one model per agent
        self.models = np.empty(self.num_users, dtype=object)
        self.target_models = np.empty(self.num_users, dtype=object)
        self.models_initial_weights = np.empty(self.num_users, dtype=object)

        for i in range(self.num_users):
            self.models[i] = DeepQNetwork(name=f'{self.model_name}_model_{i}',
                                          num_features=num_features, num_actions=len(power_level_all_channels),
                                          device=self.device)
            self.target_models[i] = DeepQNetwork(name=f'{self.model_name}_target_model_{i}',
                                                 num_features=num_features, num_actions=len(power_level_all_channels),
                                                 device=self.device)

            if self.load_pretrained_model and (self.algorithm in []):
                file = f'results/model_{i}.pt'
                self.models[i].load_checkpoint(file)

            # copy initial weights to target models
            self.models_initial_weights[i] = self.models[i].state_dict()
            self.target_models[i].load_state_dict(self.models_initial_weights[i])

        # used for updating target networks
        self.learn_step_counter = 0

        self.indexes = np.arange(self.batch_size)

        # create progress tracker
        self.T = tqdm.trange(num_episodes_train, desc='Progress', leave=True,
                             disable=bool(1 - verbose_default))

    def run_training(self, env, algorithm, saving_folder):

        rewards_history = np.zeros((num_episodes_train, num_time_slots_default))
        loss_history = np.zeros((num_episodes_train, num_time_slots_default))
        epsilon_history = np.zeros((num_episodes_train, num_time_slots_default))

        for ep in self.T:

            state, _ = env.reset(episode_num=ep, plane_this_episode=None)
            done = {i: False for i in range(self.num_users)}
            total_reward = 0

            while not any(done.values()):
                self.__update_epsilon()

                prev_state = deepcopy(state)

                action = self.make_action_for_all_users(state)

                state, reward, done, _, _ = env.step(action)
                avg_reward = np.array(list(reward.values())).mean()
                total_reward += avg_reward

                self.add_aggregated_experience_to_buffers(prev_state, state, action, reward, done)

                loss = self.train_on_random_samples()

                rewards_history[ep, env.t - 1] = avg_reward
                loss_history[ep, env.t - 1] = loss
                epsilon_history[ep, env.t - 1] = self.epsilon

            self.T.set_description(f"Reward: {(np.round(total_reward, 2))}")
            self.T.refresh()

        np.save(f'{saving_folder}/all_rewards_{algorithm}.npy', rewards_history)
        np.save(f'{saving_folder}/all_loss_values_{algorithm}.npy', loss_history)
        np.save(f'{saving_folder}/all_epsilon_values_{algorithm}.npy', epsilon_history)

        if save_model_after_train:
            for i in range(self.num_users):
                self.models[i].save_checkpoint(f'./{saving_folder}/model_{i}')

    def make_action_for_all_users(self, state, deterministic=False):

        actions = {}

        for i in range(self.num_users):
            if (np.random.random() < self.epsilon) and (not deterministic):
                actions[i] = np.random.randint(len(self.power_level_all_channels))
            else:
                observation = th.tensor(state[i], dtype=th.float).to(self.device).unsqueeze(0)
                _, advantages = self.models[i].forward(observation)

                actions[i] = th.argmax(advantages).item()

        return actions

    def add_aggregated_experience_to_buffers(self, previous_observations, new_observations,
                                             actions, rewards, dones):

        previous_observations_arr = get_numpy_from_dict_values(previous_observations)
        new_observations_arr = get_numpy_from_dict_values(new_observations)
        actions_arr = get_numpy_from_dict_values(actions)
        rewards_arr = get_numpy_from_dict_values(rewards)
        dones_arr = get_numpy_from_dict_values(dones)

        self.buffer.store_experience(
            previous_observations_arr, new_observations_arr,
            actions_arr, rewards_arr, dones_arr)

    def __update_epsilon(self, reset=False):
        if not reset:
            self.epsilon *= epsilon_decay
            self.epsilon = max(self.epsilon, epsilon_min)

        else:
            self.epsilon = epsilon_init

    def __replace_target_networks(self, model_index):
        if self.learn_step_counter == 0 \
                or (self.learn_step_counter % self.replace_target_interval) == 0:
            self.target_models[model_index].load_state_dict(self.models[model_index].state_dict())

    @staticmethod
    def __convert_value_advantage_to_q_values(v, a):
        return th.add(v, (a - a.mean(dim=1, keepdim=True)))

    def train_on_random_samples(self):

        if self.buffer.mem_counter < self.batch_size:
            return

        states, next_states, actions, reward, dones = self.buffer.sample_buffer()

        # caution: requires_grad=True is problematic on cpu
        q_predicted = th.zeros((self.batch_size, self.num_users), requires_grad=True).to(self.device)
        q_next = th.zeros((self.batch_size, self.num_users)).to(self.device)

        for i in range(self.num_users):
            # initialize local models
            self.models[i].train()
            self.models[i].optimizer.zero_grad()
            self.__replace_target_networks(model_index=i)

            V_states, A_states = self.models[i].forward(states[:, i, :])
            q_predicted[:, i] \
                = self.__convert_value_advantage_to_q_values(V_states, A_states)[self.indexes, actions[:, i]]

            with th.no_grad():
                _, A_next_states = self.models[i].forward(next_states[:, i, :])
                actions_next_states_best = A_next_states.argmax(axis=1).detach()

                V_next_states, A_next_states = self.target_models[i].forward(next_states[:, i, :])
                q_next_all_actions = self.__convert_value_advantage_to_q_values(V_next_states, A_next_states)
                q_next[:, i] = q_next_all_actions.gather(1, actions_next_states_best.unsqueeze(1)).squeeze()
                q_next[dones[:, i], i] = 0.0

        q_target = th.nan_to_num(reward).mean(axis=-1).unsqueeze(-1) + (self.gamma * q_next)

        loss = self.loss(q_predicted, q_target).to(self.device)
        loss.backward()

        for i in range(self.num_users):
            self.models[i].optimizer.step()
            self.models[i].eval()

        self.learn_step_counter += 1

        return loss.detach().cpu().numpy()

    def get_weights(self):
        return self.model.state_dict(), self.target_model.state_dict()

    def set_weights(self, weights, weights_target):
        self.model.load_state_dict(weights)
        self.target_model.load_state_dict(weights_target)

        self.model.lstm.flatten_parameters()
        self.target_model.lstm.flatten_parameters()

    def reset_models(self):
        ...
