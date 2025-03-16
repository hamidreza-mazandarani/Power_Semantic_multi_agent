import numpy as np

from config import *
from utils_area import calculate_distance

max_user_rate_per_area = max_user_rate / (min_side_size_default ** 2)


def calculate_path_loss_user_bs(user_loc, bs_loc, alpha_path_loss=2):
    # alpha_path_loss: path-loss exponent

    user_loc = np.concatenate([user_loc, [height_uav]])
    bs_loc = np.concatenate([bs_loc, [height_bs]])

    path_loss = calculate_distance(user_loc, bs_loc) ** (- alpha_path_loss)

    return path_loss


def calculate_path_losses_and_associations_all_time(user_locations_all_time, bs_locations,
                                                    num_time_slots=num_time_slots_default):
    num_users = user_locations_all_time.shape[1]
    num_base_stations = bs_locations.shape[0]

    path_losses_all_time = np.zeros((num_time_slots, num_users, num_base_stations))

    for t in range(num_time_slots):
        for u in range(num_users):
            for b in range(num_base_stations):
                path_losses_all_time[t, u, b] = calculate_path_loss_user_bs(user_locations_all_time[t, u, :],
                                                                            bs_locations[b])

    user_bs_associations_num_all_time = path_losses_all_time.argmax(axis=-1)

    return path_losses_all_time, user_bs_associations_num_all_time


def calculate_users_rates_per_channel(user_transmission_powers_one_channel,
                                      path_losses, user_bs_associations_num,
                                      consider_interference):
    num_users = user_transmission_powers_one_channel.shape[0]
    num_base_stations = path_losses.shape[1]

    users_at_bs_powers = np.tile(np.expand_dims(user_transmission_powers_one_channel, axis=1),
                                 (1, num_base_stations)) * path_losses

    bs_received_powers = users_at_bs_powers.sum(axis=0)

    users_at_selected_bs_powers = users_at_bs_powers[np.arange(num_users), user_bs_associations_num]

    if consider_interference:
        interference_per_user = [bs_received_powers[b] - users_at_selected_bs_powers[u]
                                 for u, b in enumerate(user_bs_associations_num)]
    else:
        interference_per_user = [0 for u, b in enumerate(user_bs_associations_num)]

    users_sinr = users_at_selected_bs_powers / (interference_per_user + (noise_power * np.ones(num_users)))

    users_rates = np.clip(np.log2(1 + users_sinr), None, max_user_rate)

    users_rates = np.where(users_sinr >= sinr_threshold, users_rates, 0)

    return users_rates, users_sinr


def map_rates_to_quantization_values(rates):
    period_size = max_user_rate_per_area / len(quantization_bits)

    quantization_bit_per_user = [quantization_bits[min(int(x // period_size), len(quantization_bits) - 1)]
                                 if x > 0 else -1 for x in rates]

    return quantization_bit_per_user
