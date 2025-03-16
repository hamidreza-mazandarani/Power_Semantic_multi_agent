# Environment --------------------------------------------------------------

# 'simple', 'var_num_users', 'var_num_channels', 'var_velocities'
scenario = 'simple'

num_users_default = 8
num_base_stations_default = 2
num_channels_default = 3

sinr_threshold = 0
noise_power = 1e-9

quantization_bits = [2, 4, 6, 8, 10]
num_quantization_bits = len(quantization_bits)

power_levels = [0, 5, 10]
min_power = 0
max_power = 10
max_user_rate = 10

# Plane --------------------------------------------------------------------

load_plane_setup_from_file = False

reset_plane_frequency_train = int(1e6)

plane_size = 100
placement_type = 'random'

min_side_size_default = 20
max_side_size_default = 40

min_velocity_default = 10
max_velocity_default = 20

height_uav = 100
height_bs = 50

num_planes = 6

# for calculation of segment similarities
window_size_area = 5

# 'psnr', 'ssim'
metric_type_default = 'psnr'

# Simulation ---------------------------------------------------------------
num_time_slots_default = 120

num_episodes_train = 200
num_episodes_test = 1 * num_planes

verbose_default = 0
results_folder = 'results'
save_model_after_train = True
load_pretrained_model = (num_episodes_train == 0)

seeds_test = list(range(num_episodes_test))

num_users_list = [4, 6, 8, 10]
num_channels_list = [1, 2, 3, 4]
min_velocity_list = [10, 30, 50]

# Learning -----------------------------------------------------------------

fc_sizes = [128, 64]
lstm_state_size = 256

epsilon_init = 1
epsilon_decay = 0.9995
epsilon_min = 0.001

buffer_capacity = 1000
batch_size = 64
replace_target_interval = 20
learning_rate = 0.001
history_size = 4
gamma = 0.75
