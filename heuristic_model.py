import numpy as np

from config import max_power


class HeuristicModel:
    def __init__(self, num_users, num_base_stations, num_channels, power_level_all_channels, algorithm):

        self.num_users = num_users
        self.num_base_stations = num_base_stations
        self.num_channels = num_channels
        self.power_level_all_channels = power_level_all_channels
        self.algorithm = algorithm

    def calculate_action_for_selected_users(self, users_to_send):
        actions = {}
        channel_counter = 0

        for u in np.arange(self.num_users):

            power_level = np.zeros(self.num_channels)

            if u in users_to_send:
                power_level[channel_counter % self.num_channels] = max_power
                channel_counter += 1

                actions[u] = self.power_level_all_channels.index(tuple(power_level))

        return actions

    def make_action(self, path_losses_this_time):

        if self.algorithm == 'heuristic_c':

            best_path_loss_this_time = path_losses_this_time.max(axis=1)
            users_to_send = np.argpartition(best_path_loss_this_time, -self.num_channels)[-self.num_channels:]

            actions = self.calculate_action_for_selected_users(users_to_send)

            for u in np.arange(self.num_users):
                if u not in actions.keys():
                    actions[u] = 0

        elif self.algorithm == 'heuristic_d':

            actions = self.calculate_action_for_selected_users(users_to_send=np.arange(self.num_users))

        else:
            raise NotImplementedError()

        return actions
