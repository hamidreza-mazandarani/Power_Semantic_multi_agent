import gymnasium
import matplotlib.pyplot as plt
from gymnasium import spaces

from utils_area import *
from utils_network import *


class Env(gymnasium.Env):
    def __init__(self, algorithm,
                 num_users=num_users_default, num_base_stations=num_base_stations_default,
                 num_time_slots=num_time_slots_default, num_channels=num_channels_default,
                 min_velocity=min_velocity_default, max_velocity=max_velocity_default,
                 debug_time_slots=()):  # 10, 60, 110
        super(Env, self).__init__()

        self.algorithm = algorithm
        self.num_users = num_users
        self.num_base_stations = num_base_stations
        self.num_time_slots = num_time_slots
        self.num_channels = num_channels
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.debug_time_slots = debug_time_slots

        # region area-related variables
        self.all_segments = make_all_segments(self.num_users)

        self.segments_corners = np.zeros((num_time_slots, len(self.all_segments), 4))

        self.planes = []

        for plane in range(num_planes):
            this_plane = np.load(f'planes/image_{plane}.npy').astype(float)

            assert this_plane.shape[0] == this_plane.shape[1]  # image size
            assert this_plane.shape[-1] == len(quantization_bits) + 1  # quality of image

            self.plane_size = this_plane.shape[0]
            self.num_colors = this_plane.shape[-2]

            self.planes.append(this_plane)

        # the SSIM of each segment for each quality
        self.similarities = np.zeros((self.num_time_slots, len(self.all_segments), num_quantization_bits))
        # endregion

        # actions: determine power level
        self.power_level_all_channels = [x for x in itertools.product(power_levels, repeat=self.num_channels)
                                         if min_power <= sum(x) <= max_power]

        self.action_space = {i: spaces.Discrete(len(self.power_level_all_channels)) for i in range(self.num_users)}

        if self.algorithm == 'BO':
            self.num_features = (self.num_base_stations  # path loss to each BS
                                 + 2  # UAV location
                                 )
        else:
            self.num_features = (self.num_base_stations  # path loss to each BS
                                 + 2  # location
                                 + 1  # avg shared coverage
                                 )

        self.observation_space = {i: spaces.Box(low=0, high=1,
                                                shape=(history_size, self.num_features),
                                                dtype=np.float32)
                                  for i in range(self.num_users)}

        self.observations = {i: np.zeros((history_size, self.num_features)) for i in range(self.num_users)}

        self.t = 0

        self.user_locations_all_time = np.zeros((self.num_time_slots, self.num_users, 2))
        self.users_corners_all_time = np.zeros((self.num_time_slots, self.num_users, 4))
        self.base_stations_locations = locate_base_stations(self.num_base_stations)
        self.sides_arr = np.zeros(self.num_users)

        self.path_losses_all_time = np.zeros((self.num_time_slots, self.num_users, self.num_base_stations))
        self.user_bs_associations_num_all_time = np.zeros((self.num_time_slots, self.num_users), dtype=int)

        self.user_transmission_powers_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.rates_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.sinrs_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))

        self.qualities_all_time = np.zeros((self.num_time_slots, plane_size, plane_size))
        self.observed_pixels_all_time = np.zeros((self.num_time_slots, plane_size, plane_size))

        self.features_history = np.zeros((self.num_time_slots, self.num_users, self.num_features))
        self.quantization_level_all_times = np.zeros((self.num_time_slots, self.num_users))

    def reset(self, **kwargs):
        self.t = 0
        episode_num = kwargs['episode_num']
        plane_this_episode = kwargs['plane_this_episode']

        if plane_this_episode is None:
            plane_this_episode = np.random.randint(num_planes)
        # print(f'Loading Plane #{plane_this_episode}')

        self.user_transmission_powers_all_time = np.zeros((self.num_time_slots, self.num_channels,
                                                           self.num_users))
        self.rates_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.sinrs_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))

        self.qualities_all_time = np.zeros((self.num_time_slots, plane_size, plane_size))
        self.observed_pixels_all_time = np.zeros((self.num_time_slots, plane_size, plane_size))

        if (episode_num == 0) or (episode_num % reset_plane_frequency_train == 0):

            if load_plane_setup_from_file:
                self.user_locations_all_time = np.load('topos/user_locations_all_time.npy')
                assert self.user_locations_all_time.shape == (self.num_time_slots, self.num_users, 2)

                self.sides_arr = np.load('topos/sides_arr.npy')
                assert self.sides_arr.shape == (self.num_users,)

                self.segments_corners = np.load('topos/segments_corners.npy')
                assert self.segments_corners.shape == (self.num_time_slots, len(self.all_segments), 4)

                self.users_corners_all_time = np.load('topos/users_corners_all_time.npy')
                assert self.users_corners_all_time.shape == (self.num_time_slots, self.num_users, 4)

                self.base_stations_locations = np.load('topos/bs_locations.npy')
                assert self.base_stations_locations.shape == (self.num_base_stations, 2)

                self.similarities = np.load(f'topos/similarities_{plane_this_episode}.npy')
                assert self.similarities.shape == (self.num_time_slots, len(self.all_segments),
                                                   num_quantization_bits)

            else:
                self.user_locations_all_time = determine_users_locations(num_users=self.num_users,
                                                                         num_time_slots=self.num_time_slots,
                                                                         min_velocity=self.min_velocity,
                                                                         max_velocity=self.max_velocity)

                self.segments_corners, self.users_corners_all_time, self.sides_arr \
                    = determine_segments_all_time(self.all_segments, self.user_locations_all_time)

                self.similarities = calculate_quality_values(self.all_segments, self.segments_corners,
                                                             self.planes[plane_this_episode])

            self.base_stations_locations = locate_base_stations(self.num_base_stations)

            self.path_losses_all_time, self.user_bs_associations_num_all_time \
                = calculate_path_losses_and_associations_all_time(self.user_locations_all_time,
                                                                  self.base_stations_locations)

        # Reset the state of the environment to an initial state
        self.observations = {i: np.zeros((history_size, self.num_features)) for i in range(self.num_users)}

        self.similarities = calculate_quality_values(self.all_segments, self.segments_corners,
                                                     self.planes[0])

        return self.observations, {}

    def step(self, action_dict):
        # Execute one time step within the environment

        # Segment Quality Calculations --------------------------------------------------------------------------------
        self.user_transmission_powers_all_time[self.t, :, :] = np.array([list(self.power_level_all_channels[a])
                                                                         for a in action_dict.values()]).T

        for c in range(self.num_channels):
            self.rates_all_time[self.t, c, :], self.sinrs_all_time[self.t, c, :] = \
                calculate_users_rates_per_channel(self.user_transmission_powers_all_time[self.t, c, :],
                                                  self.path_losses_all_time[self.t, :],
                                                  self.user_bs_associations_num_all_time[self.t, :],
                                                  consider_interference=(self.algorithm != 'Optimal'))

        rate_per_user_this_time = self.rates_all_time[self.t, :, :].sum(axis=0)
        rate_per_user_this_time_per_area = rate_per_user_this_time / (self.sides_arr ** 2)
        quantization_level_per_user = map_rates_to_quantization_values(rate_per_user_this_time_per_area)

        self.quantization_level_all_times[self.t] = quantization_level_per_user

        active_segments_this_time = np.nonzero(self.segments_corners[self.t].sum(axis=-1) > 0)[0]

        # active_nodes = np.nonzero(rate_per_user_this_time_slot)[0]

        quality_per_pixel_this_time = np.zeros((plane_size, plane_size))
        observed_pixels_this_time = np.full((plane_size, plane_size), np.nan)
        segment_qualities_dict = {}

        for segment_index in active_segments_this_time:

            if self.algorithm == 'optimal':
                segment_max_quantization_level = max(quantization_bits)
            else:
                segment_members = self.all_segments[segment_index]
                segment_max_quantization_level = max([quantization_level_per_user[x] for x in segment_members])

                if (segment_max_quantization_level <= 0) and (len(segment_members) > 1):
                    # joint segments with more than one member,
                    # can be excluded if they do not provide better quality
                    continue

            if segment_max_quantization_level > 0:
                segment_quality = self.similarities[
                    self.t, segment_index, quantization_bits.index(segment_max_quantization_level)]
            else:
                segment_quality = 0

            segment_qualities_dict[segment_index] = segment_quality

            segment_corners = self.segments_corners[self.t, segment_index]

            selection_index = get_segment_selection_index(segment_corners)

            if selection_index is None:
                continue

            observed_pixels_this_time[selection_index] = 1

            quality_per_pixel_this_time[selection_index] = np.maximum(quality_per_pixel_this_time[selection_index],
                                                                      segment_quality)

        self.qualities_all_time[self.t] = quality_per_pixel_this_time
        self.observed_pixels_all_time[self.t] = observed_pixels_this_time

        # Reward Calculations -----------------------------------------------------------------------------------------
        if self.algorithm == 'BO':
            reward = {i: rate_per_user_this_time.mean() / max_user_rate for i in range(self.num_users)}
        else:
            quality_in_observed_pixels = quality_per_pixel_this_time[observed_pixels_this_time == 1]
            avg_quality_in_observed_pixels = quality_in_observed_pixels.mean() \
                if len(quality_in_observed_pixels) > 0 else 0
            reward = {i: avg_quality_in_observed_pixels for i in range(self.num_users)}

        # Debugging ---------------------------------------------------------------------------------------------------
        if self.t in self.debug_time_slots:
            print('actions                      :\n', self.user_transmission_powers_all_time[self.t])
            print('active segments              :', active_segments_this_time)
            print('rate per user                :', rate_per_user_this_time)
            print('Quantization level per user  :', quantization_level_per_user)
            print('Segment qualities            :', segment_qualities_dict)
            print('reward                       :', list(reward.values()))

            fig, ax = plt.subplots(1)
            im = ax.imshow(np.transpose(self.qualities_all_time[self.t]), origin='lower', cmap='Greys')
            ax.set_title(f'Qualities at {self.t}')
            for x_ind, x in enumerate(self.segments_corners[self.t, active_segments_this_time]):
                if x_ind < self.num_users:
                    # ignore single-user segments
                    continue
                rectangle = plt.Rectangle([x[3], x[2]],
                                          (x[1] - x[3]), (x[0] - x[2]),
                                          edgecolor='orange', facecolor='none', linewidth=2)
                ax.add_patch(rectangle)
            plt.colorbar(im, ax=ax)
            plt.show()

        # Next State Calculations -------------------------------------------------------------------------------------

        done = {i: bool(self.t >= self.num_time_slots - 2) for i in range(self.num_users)}

        self.t += 1

        active_segments_next_time = np.nonzero(self.segments_corners[self.t].sum(axis=-1) > 0)[0]

        pixels_observation_level_next_time = np.zeros((plane_size, plane_size))
        for segment_index in active_segments_next_time:
            segment_members = self.all_segments[segment_index]

            selection_index = get_segment_selection_index(self.segments_corners[self.t, segment_index])

            if (selection_index is not None) and (len(segment_members) > 1):
                pixels_observation_level_next_time[selection_index] += 1

        user_locations_normalized = (self.user_locations_all_time[self.t] / plane_size)

        # making local observations
        for i in range(self.num_users):

            path_loss_this_user_clipped = np.clip(self.path_losses_all_time[self.t, i, :].flatten(), 0, 10)

            user_shared_coverage = (calculate_user_shared_coverage(pixels_observation_level_next_time,
                                                                   self.users_corners_all_time[self.t, i])
                                    / len(active_segments_next_time))

            self.observations[i][:-1, :] = self.observations[i][1:, :]

            if self.algorithm == 'BO':
                self.observations[i][-1, :] = np.concatenate([path_loss_this_user_clipped,
                                                              user_locations_normalized[i]])
            else:
                self.observations[i][-1, :] = np.concatenate([path_loss_this_user_clipped,
                                                              user_locations_normalized[i],
                                                              [user_shared_coverage]])

            self.features_history[self.t, i] = self.observations[i][-1, :]

        return self.observations, reward, done, done, {}
