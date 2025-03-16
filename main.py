import numpy as np

from config import *
from simulation_runner import SimulationRunner

if scenario == 'simple':

    num_users = num_users_default
    num_base_stations = num_base_stations_default
    num_time_slots = num_time_slots_default
    num_channels = num_channels_default
    min_velocity = min_velocity_default
    max_velocity = max_velocity_default
    np.save(f'{results_folder}/scenario_{scenario}_config_{0}.npy',
            np.array([num_users, num_base_stations, num_time_slots, num_channels]))

    # heuristic_c: centralized heuristic, heuristic_d: distributed heuristic
    algorithms = {'optimal': 0, 'heuristic_c': 0, 'heuristic_d': 1, 'BO': 0, 'TO': 0}

    for algo, is_active in algorithms.items():

        if not is_active:
            continue

        simulation_runner = SimulationRunner(algo, num_users, num_base_stations, num_time_slots, num_channels,
                                             min_velocity, max_velocity,
                                             name=f'simple')

        simulation_runner.run_one_episode()

elif scenario == 'var_num_users':
    for num_users in num_users_list:
        num_base_stations = num_base_stations_default
        num_time_slots = num_time_slots_default
        num_channels = num_channels_default
        min_velocity = min_velocity_default
        max_velocity = max_velocity_default
        np.save(f'{results_folder}/scenario_{scenario}_config_{num_users}.npy',
                np.array([num_users, num_base_stations, num_time_slots, num_channels]))

        algorithms = {'optimal': 0, 'heuristic_c': 0, 'heuristic_d': 1, 'BO': 0, 'TO': 0}

        for algo, is_active in algorithms.items():

            if not is_active:
                continue

            simulation_runner = SimulationRunner(algo,
                                                 num_users, num_base_stations, num_time_slots, num_channels,
                                                 min_velocity, max_velocity,
                                                 name=f'num_users_{num_users}'
                                                 )

            simulation_runner.run_one_episode()

elif scenario == 'var_num_channels':
    for num_channels in num_channels_list:
        num_users = num_users_default
        num_base_stations = num_base_stations_default
        num_time_slots = num_time_slots_default
        min_velocity = min_velocity_default
        max_velocity = max_velocity_default
        np.save(f'{results_folder}/scenario_{scenario}_config_{num_channels}.npy',
                np.array([num_users, num_base_stations, num_time_slots, num_channels]))

        algorithms = {'optimal': 0, 'heuristic_c': 0, 'heuristic_d': 1, 'BO': 0, 'TO': 0}

        for algo, is_active in algorithms.items():

            if not is_active:
                continue

            simulation_runner = SimulationRunner(algo,
                                                 num_users, num_base_stations, num_time_slots, num_channels,
                                                 min_velocity, max_velocity,
                                                 name=f'num_channels_{num_channels}')

            simulation_runner.run_one_episode()

elif scenario == 'var_velocities':
    for min_velocity in min_velocity_list:
        num_users = num_users_default
        num_base_stations = num_base_stations_default
        num_time_slots = num_time_slots_default
        num_channels = num_channels_default
        max_velocity = min_velocity + 10
        np.save(f'{results_folder}/scenario_{scenario}_config_{min_velocity}.npy',
                np.array([num_users, num_base_stations, num_time_slots, num_channels]))

        algorithms = {'optimal': 0, 'heuristic_c': 0, 'heuristic_d': 1, 'BO': 0, 'TO': 0}

        for algo, is_active in algorithms.items():

            if not is_active:
                continue

            simulation_runner = SimulationRunner(algo,
                                                 num_users, num_base_stations, num_time_slots, num_channels,
                                                 min_velocity, max_velocity,
                                                 name=f'min_velocities_{min_velocity}')

            simulation_runner.run_one_episode()
