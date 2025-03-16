import os

import numpy as np
import torch

from config import *
from env import Env
from heuristic_model import HeuristicModel
from ma_d3ql_model import MA_D3QL


def make_dir(folder_name, parent):
    if not (folder_name in os.listdir(parent)):
        os.makedirs(parent + '/' + folder_name)


class SimulationRunner:

    def __init__(self, algorithm, num_users, num_base_stations, num_time_slots, num_channels,
                 min_velocity, max_velocity,
                 name=None):
        self.algorithm = algorithm

        self.num_users = num_users
        self.num_base_stations = num_base_stations
        self.num_time_slots = num_time_slots
        self.num_channels = num_channels
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.name = name

        self.env = Env(algorithm=self.algorithm,
                       num_users=self.num_users, num_base_stations=self.num_base_stations,
                       num_channels=self.num_channels,
                       min_velocity=self.min_velocity, max_velocity=self.max_velocity)

        if self.name is not None:
            self.saving_folder = results_folder + '/' + self.name
            make_dir(self.name, results_folder)
        else:
            self.saving_folder = results_folder

    def run_one_episode(self):
        print(f'starting algorithm {self.algorithm} for scenario {self.name}...')

        if self.algorithm in ['heuristic_c', 'heuristic_d']:
            model_heuristic = HeuristicModel(self.num_users, self.num_base_stations,
                                             self.env.num_channels,
                                             self.env.power_level_all_channels,
                                             algorithm=self.algorithm)

        elif self.algorithm == 'optimal':
            pass

        else:
            ma_d3ql = MA_D3QL(self.num_users, self.env.num_channels, self.env.power_level_all_channels,
                              num_features=self.env.num_features,
                              algorithm=self.algorithm)

            ma_d3ql.run_training(self.env, algorithm=self.algorithm, saving_folder=self.saving_folder)

        for ep in range(num_episodes_test):

            torch.manual_seed(seeds_test[ep])
            np.random.seed(seeds_test[ep])

            # Test environment
            # we set episode_num to zero for a new plane with each reset
            state, _ = self.env.reset(episode_num=0, plane_this_episode=ep % num_planes)
            done = {i: False for i in range(self.num_users)}

            while not any(done.values()):
                if self.algorithm == 'optimal':
                    action = {i: len(self.env.power_level_all_channels) - 1  # maximum power on one channel
                              for i in range(self.num_users)}
                elif self.algorithm in ['heuristic_c', 'heuristic_d']:
                    action = model_heuristic.make_action(
                        path_losses_this_time=self.env.path_losses_all_time[self.env.t])
                else:
                    action = ma_d3ql.make_action_for_all_users(state, deterministic=True)

                state, reward, done, _, _ = self.env.step(action)

            print(f"Total Quality: {self.env.qualities_all_time.mean()}")

            np.save(f'{self.saving_folder}/user_locations_all_time_{self.algorithm}_{ep}.npy',
                    self.env.user_locations_all_time)
            np.save(f'{self.saving_folder}/bs_locations_{self.algorithm}_{ep}.npy',
                    self.env.base_stations_locations)
            np.save(f'{self.saving_folder}/sides_users_{self.algorithm}_{ep}.npy',
                    self.env.sides_arr.astype(float))
            np.save(f'{self.saving_folder}/user_bs_associations_num_all_time_{self.algorithm}_{ep}.npy',
                    self.env.user_bs_associations_num_all_time)
            np.save(f'{self.saving_folder}/rates_all_time_{self.algorithm}_{ep}.npy',
                    self.env.rates_all_time)
            np.save(f'{self.saving_folder}/segments_corners_arr_{self.algorithm}_{ep}.npy',
                    self.env.segments_corners.astype(float))
            np.save(f'{self.saving_folder}/qualities_all_time_{self.algorithm}_{ep}.npy',
                    self.env.qualities_all_time)
            np.save(f'{self.saving_folder}/observed_pixels_all_time_{self.algorithm}_{ep}.npy',
                    self.env.observed_pixels_all_time)
            np.save(f'{self.saving_folder}/user_transmission_powers_all_time_{self.algorithm}_{ep}.npy',
                    self.env.user_transmission_powers_all_time)
            np.save(f'{self.saving_folder}/features_history_{self.algorithm}_{ep}.npy',
                    self.env.features_history)
            np.save(f'{self.saving_folder}/quantization_level_all_times_{self.algorithm}_{ep}.npy',
                    self.env.quantization_level_all_times)

        self.env.close()
