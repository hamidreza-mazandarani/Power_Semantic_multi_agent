import itertools

import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio

from config import *

np.seterr(divide='ignore', invalid='ignore')


def locate_base_stations(num_base_stations, is_random=False):
    if is_random:
        bs_locations = np.random.uniform(plane_size, size=(num_base_stations, 2))
    else:
        bs_locations = np.concatenate([[np.linspace(0, plane_size, num=num_base_stations + 2)[1:-1]],
                                       [plane_size / 2 * np.ones(num_base_stations).T]],
                                      axis=0).T
    return bs_locations


def calculate_distance(node_1_pos, node_2_pos):
    return np.sqrt(((node_1_pos - node_2_pos) ** 2).sum())


def generate_circle_points(x, y, r, num_points=100, random_direction=True):
    # Center of the circle (make sure it fits inside the rectangle)
    center_x = np.random.uniform(r, x - r)
    center_y = np.random.uniform(r, y - r)

    # Generate direction of the movement (clockwise or counter-clockwise)
    if random_direction:
        direction = np.random.choice([0, 1])
    else:
        direction = int(center_x < (x / 2))

    # Generate angle values
    if direction:
        theta = np.linspace(0, 2 * np.pi, num_points)
    else:
        theta = np.linspace(2 * np.pi, 0, num_points)

    # Calculate points on the circumference
    circle_points = [
        (center_x + r * np.cos(angle), center_y + r * np.sin(angle))
        for angle in theta
    ]

    return np.array(circle_points)


def make_all_segments(num_users):
    all_segments = []

    for subset_size in range(1, num_users + 1):
        for subset in itertools.combinations(np.arange(num_users), subset_size):
            all_segments.append(subset)

    return all_segments


def determine_users_locations(num_users=num_users_default, num_time_slots=num_time_slots_default,
                              min_velocity=min_velocity_default, max_velocity=max_velocity_default,
                              ):
    velocities = np.random.uniform(min_velocity, max_velocity, size=num_users)

    user_locations_all_time = np.zeros((num_time_slots, num_users, 2))

    for u in range(num_users):
        user_locations_all_time[:, u, :] = \
            generate_circle_points(plane_size, plane_size, velocities[u], num_points=num_time_slots)

    return user_locations_all_time


def determine_segments_all_time(all_segments, user_locations_all_time,
                                num_time_slots=num_time_slots_default,
                                min_side_size=min_side_size_default, max_side_size=max_side_size_default,
                                ):
    num_users = user_locations_all_time.shape[1]

    sides_arr = np.random.uniform(min_side_size, max_side_size, size=num_users)

    users_corners_all_time = find_users_corners(user_locations_all_time, sides_arr, num_time_slots)

    segments_corners = np.zeros((num_time_slots, len(all_segments), 4))

    for t in range(num_time_slots):
        for x_ind, x in enumerate(all_segments):
            segments_corners[t, x_ind] = calculate_common_segment(users_corners_all_time[t, list(x)])

    return segments_corners, users_corners_all_time, sides_arr


def find_users_corners(user_locations_all_time, sides_arr, num_time_slots):
    num_users = user_locations_all_time.shape[1]

    # up, right, down, left
    users_corners_all_time = np.zeros((num_time_slots, num_users, 4))

    for t in range(num_time_slots):
        for u in range(num_users):
            users_corners_all_time[t, u, 0] = user_locations_all_time[t, u, 1] + (sides_arr[u] / 2)
            users_corners_all_time[t, u, 1] = user_locations_all_time[t, u, 0] + (sides_arr[u] / 2)
            users_corners_all_time[t, u, 2] = user_locations_all_time[t, u, 1] - (sides_arr[u] / 2)
            users_corners_all_time[t, u, 3] = user_locations_all_time[t, u, 0] - (sides_arr[u] / 2)

    users_corners_all_time = np.clip(users_corners_all_time, 0, plane_size)

    return users_corners_all_time


def calculate_common_segment(rectangles):
    # Initialize the overlapping area with the first rectangle
    y_top = rectangles[:, 0].min()
    x_right = rectangles[:, 1].min()
    y_bottom = rectangles[:, 2].max()
    x_left = rectangles[:, 3].max()

    # Check for overlap
    if x_left < x_right and y_bottom < y_top:
        return np.array([y_top, x_right, y_bottom, x_left])  # Return corners of the common area
    else:
        return np.array([0, 0, 0, 0])  # No common area exists


def calculate_quality_values(all_segments, segments_corners, image_np,
                             num_time_slots=num_time_slots_default, window_size=window_size_area,
                             metric_type=metric_type_default, normalize=True):
    quality_values = np.zeros((num_time_slots, len(all_segments), num_quantization_bits))

    for t in range(num_time_slots):
        for segment_ind, _ in enumerate(all_segments):
            if segments_corners[t, segment_ind].sum() > 0:
                this_segment_corners = segments_corners[t, segment_ind]
                selection_h = np.arange(int(this_segment_corners[3]), int(this_segment_corners[1]))
                selection_v = np.arange(int(this_segment_corners[2]), int(this_segment_corners[0]))

                if len(selection_h) < window_size or len(selection_v) < window_size:
                    continue

                for q_ind, q in enumerate(quantization_bits):
                    if metric_type == 'ssim':
                        similarity_index = ssim(image_np[np.ix_(selection_h, selection_v)][:, :, :, 0],
                                                image_np[np.ix_(selection_h, selection_v)][:, :, :, q_ind + 1],
                                                channel_axis=2, win_size=window_size)
                    elif metric_type == 'psnr':
                        similarity_index = peak_signal_noise_ratio(
                            image_np[np.ix_(selection_h, selection_v)][:, :, :, 0],
                            image_np[np.ix_(selection_h, selection_v)][:, :, :, q_ind + 1],
                            data_range=255)
                    else:
                        raise NotImplementedError()

                    quality_values[t, segment_ind, q_ind] = similarity_index

    if normalize:
        # normalize to best quality for each segment at each time slot
        quality_values /= quality_values.max(axis=-1, keepdims=True)

        quality_values = np.nan_to_num(quality_values)

    return quality_values


def get_segment_selection_index(segment_corners):
    selection_h = np.arange(int(segment_corners[3]), int(segment_corners[1]))
    selection_v = np.arange(int(segment_corners[2]), int(segment_corners[0]))

    if len(selection_h) < window_size_area or len(selection_v) < window_size_area:
        return None

    selection_index = np.ix_(selection_h, selection_v)

    return selection_index


def calculate_user_shared_coverage(pixels_observation_level, user_corners_this_time):
    selection_h = np.arange(int(user_corners_this_time[3]), int((user_corners_this_time[1])))
    selection_v = np.arange(int((user_corners_this_time[2])), int((user_corners_this_time[0])))
    selection_index = np.ix_(selection_h, selection_v)
    avg_observation_level_this_user = pixels_observation_level[selection_index].mean()

    return avg_observation_level_this_user


def calculate_segment_size(this_segment_corners):
    assert this_segment_corners[0] >= this_segment_corners[2], 'up < bottom'
    assert this_segment_corners[1] >= this_segment_corners[3], 'right < left'

    return ((this_segment_corners[0] - this_segment_corners[2])
            * (this_segment_corners[1] - this_segment_corners[3]))

# # Circle-based Calculations -----------------------------------------------------------------------------------------
# def calculate_distances_all_users(user_locations):
#     num_users = user_locations.shape[0]
#
#     distances = np.zeros((num_users, num_users))
#
#     for i in range(num_users):
#         for j in range(num_users):
#             if j == i:
#                 distances[i, j] = 0.0
#             if j > i:
#                 distances[i, j] = calculate_distance(user_locations[i, :],
#                                                      user_locations[j, :])
#             else:
#                 distances[i, j] = distances[j, i]
#
#     return distances
#
#
# def calculate_mutual_area_two_circles(r1, r2, d):
#     # No overlap
#     if d >= r1 + r2:
#         return 0.0
#
#     # One circle inside the other
#     if d <= abs(r1 - r2):
#         return min(np.pi * r1 ** 2,
#                    np.pi * r2 ** 2)
#
#     # Partial overlap
#     r1_sq = r1 ** 2
#     r2_sq = r2 ** 2
#
#     part_1 = r1_sq * np.arccos((d ** 2 + r1_sq - r2_sq) / (2 * d * r1))
#     part_2 = r2_sq * np.arccos((d ** 2 + r2_sq - r1_sq) / (2 * d * r2))
#     part_3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
#
#     # Total mutual area
#     return part_1 + part_2 - part_3
#
#
# def calculate_mutual_area_two_squares(r1, r2, d):
#     # No overlap
#     if d >= r1 + r2:
#         return 0.0
#
#     # One circle inside the other
#     if d <= abs(r1 - r2):
#         return min(np.pi * r1 ** 2,
#                    np.pi * r2 ** 2)
#
#     # Partial overlap
#     r1_sq = r1 ** 2
#     r2_sq = r2 ** 2
#
#     part_1 = r1_sq * np.arccos((d ** 2 + r1_sq - r2_sq) / (2 * d * r1))
#     part_2 = r2_sq * np.arccos((d ** 2 + r2_sq - r1_sq) / (2 * d * r2))
#     part_3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
#
#     # Total mutual area
#     return part_1 + part_2 - part_3
#
#
# def calculate_mutual_area_all_users(radius_arr, distances):
#     num_users = radius_arr.shape[0]
#
#     mutual_areas = np.zeros((num_users, num_users))
#
#     for i in range(num_users):
#         for j in range(num_users):
#             if j == i:
#                 mutual_areas[i, j] = np.pi * (radius_arr[i] ** 2)
#             if j > i:
#                 mutual_areas[i, j] = calculate_mutual_area_two_circles(radius_arr[i], radius_arr[j],
#                                                                        distances[i, j])
#             else:
#                 mutual_areas[i, j] = mutual_areas[j, i]
#
#     return mutual_areas
#
#
# def calculate_higher_level_mutual_areas(segment, user_locations, radius_arr):
#     x_range = np.arange(np.floor(user_locations[segment, 0].min()).astype(int),
#                         np.ceil(user_locations[segment, 0].max()).astype(int))
#
#     y_range = np.arange(np.floor(user_locations[segment, 1].min()).astype(int),
#                         np.ceil(user_locations[segment, 1].max()).astype(int))
#
#     area = 0
#
#     for x in x_range:
#         for y in y_range:
#             area += int(all(calculate_distance([x, y], user_locations[node, :] <= radius_arr[node])
#                             for node in segment))
#
#     return area
#
#
# def calculate_segment_areas(segments_list, mutual_areas, user_locations, radius_arr):
#     segment_areas = {}
#
#     for segment in segments_list:
#         if len(segment) == 1:
#             segment_areas[tuple(segment)] = mutual_areas[segment[0], segment[0]]
#         elif len(segment) == 2:
#             segment_areas[tuple(segment)] = mutual_areas[segment[0], segment[1]]
#         else:
#             segment_areas[tuple(segment)] = calculate_higher_level_mutual_areas(segment, user_locations, radius_arr)
#
#     return segment_areas
#
#
# distances_between_users_all_times = np.full((num_time_slots, num_users, num_users), np.nan)
# mutual_areas_all_times = np.full((num_time_slots, num_users, num_users), np.nan)
# segments_area_df = pd.DataFrame(columns=[str(x) for x in all_segments],
#                                index=np.arange(num_time_slots))
# area_corners = np.zeros(())
# for t in range(num_time_slots):
#     distances_between_users_all_times[t, :, :] = calculate_distances_all_users(
#         user_locations_all_time[t, :, :])
#
#     mutual_areas_all_times[t, :, :] = calculate_mutual_area_all_users(sides_arr,
#                                                                       distances_between_users_all_times[t, :, :])
#
#     have_mutual_areas = (mutual_areas_all_times[t, :, :] > 0).astype(int)
#
#     have_mutual_areas_graph = nx.from_numpy_array(have_mutual_areas)
#
#     segments_list = list(nx.enumerate_all_segments(have_mutual_areas_graph))
#
#     segment_areas = calculate_segment_areas(segments_list, mutual_areas_all_times[t, :, :],
#                                           user_locations_all_time[t, :, :], sides_arr)
#
#     segment_areas_dict = {str(k): segment_areas[k] if k in segment_areas else 0 for k in all_segments}
