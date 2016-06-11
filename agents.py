from qengine import *


def initialize_doom(config_file, grayscale=True):
    doom = DoomGame()
    doom.load_config("common.cfg")
    doom.load_config(config_file)
    if grayscale:
        doom.set_screen_format(ScreenFormat.GRAY8)
    print "Initializing DOOM ..."
    doom.init()
    print "DOOM initialized."
    return doom


def predict():
    game = initialize_doom("config/predict_position.cfg")
    network_args = {
        "ddqn": True
    }
    engine_args = {
        "reshaped_x": 84,
        "reshaped_y": 84,
        "remember_n_actions": 4,
        "skiprate": 3,
        "name": "predict2_duelling",
        "net_type": "duelling",
        "melt_steps": 10000,
        "epsilon_decay_steps": 500000,
        "epsilon_decay_start_step": 0,
        "end_epsilon": 0.005,
        "start_epsilon": 1.0,
        "update_pattern": (4, 1),
        "backprop_start_step": 10000,
        "replay_memory_size": 20000
    }
    return QEngine(game=game, network_args=network_args, **engine_args)


def predict_supreme():
    game = initialize_doom("config/predict_position_supreme.cfg")
    network_args = {
        "ddqn": True,
        # "gamma": 1
    }
    engine_args = {
        "name": "predict_s_dueling_count",
        "net_type": "dueling",
        "reshaped_x": 100,
        "reshaped_y": 75,
        "skiprate": 3,

        "remember_n_actions": 4,
        "count_states": True,
        # "use_game_variables": True,
        "shaping_on": False,

        "melt_steps": 10000,
        "epsilon_decay_steps": 500000,
        "epsilon_decay_start_step": 0,
        "start_epsilon": 1.0,
        "end_epsilon": 0.005,

        "update_pattern": (4, 1),
        "backprop_start_step": 10000,
        "replay_memory_size": 20000,
        # "misc_scale": [3/2100.0]
    }
    return QEngine(game=game, network_args=network_args, **engine_args)


def health_supreme():
    game = initialize_doom("config/health_gathering_supreme.cfg")
    network_args = {
        "ddqn": True,
        "gamma": 1
    }
    engine_args = {
        "name": "health_s_dueling_skip3_miscscale_x100_gamma1",
        "net_type": "dueling",
        "reshaped_x": 100,
        "reshaped_y": 75,
        "skiprate": 3,

        "remember_n_actions": 4,
        "count_states": True,
        "use_game_variables": True,
        "shaping_on": True,

        "melt_steps": 10000,
        "epsilon_decay_steps": 500000,
        "epsilon_decay_start_step": 0,
        "start_epsilon": 1.0,
        "end_epsilon": 0.005,

        "update_pattern": (4, 1),
        "backprop_start_step": 10000,
        "replay_memory_size": 20000,
        "misc_scale": [0.01, 7 / 2100.0]
    }
    return QEngine(game=game, network_args=network_args, **engine_args)


def health():
    game = initialize_doom("config/health_gathering.cfg")
    network_args = {
        "ddqn": True,
        "gamma": 1
    }
    engine_args = {
        "name": "health_noshaping_onehot_nocount",
        "net_type": "dueling",
        "reshaped_x": 100,
        "reshaped_y": 75,
        "skiprate": 7,

        "remember_n_actions": 4,
        "one_hot": True,
        "count_states": False,
        "use_game_variables": True,
        "shaping_on": False,

        "melt_steps": 10000,
        "epsilon_decay_steps": 500000,
        "epsilon_decay_start_step": 0,
        "start_epsilon": 1.0,
        "end_epsilon": 0.005,

        "update_pattern": (4, 1),
        "backprop_start_step": 10000,
        "replay_memory_size": 20000,
        # "misc_scale": [0.01, 7/2100.0]
    }
    return QEngine(game=game, network_args=network_args, **engine_args)


def defend_the_center():
    game = initialize_doom("config/defend_the_center.cfg")
    network_args = {
        "ddqn": True
    }
    engine_args = {
        "name": "center_up4_dueling",
        "net_type": "dueling",
        "reshaped_x": 84,
        "reshaped_y": 84,
        "skiprate": 3,

        "remember_n_actions": 4,
        "count_states": False,
        "use_game_variables": True,

        "melt_steps": 10000,
        "epsilon_decay_steps": 500000,
        "epsilon_decay_start_step": 0,
        "start_epsilon": 1.0,
        "end_epsilon": 0.005,

        "update_pattern": (4, 1),
        "backprop_start_step": 10000,
        "replay_memory_size": 10000,
    }
    return QEngine(game=game, network_args=network_args, **engine_args)


def pacman():
    game = initialize_doom("config/pacman.cfg")
    network_args = {
        "ddqn": True,
        "gamma": 1.0
    }
    engine_args = {
        "name": "pacman_dueling_gamma1_one_hot",
        "net_type": "dueling",
        "reshaped_x": 100,
        "reshaped_y": 75,
        "skiprate": 3,

        "remember_n_actions": 4,
        "one_hot": True,
        "count_states": True,
        "use_game_variables": True,
        # "shaping_on": True,

        "melt_steps": 10000,
        "epsilon_decay_steps": 500000,
        "epsilon_decay_start_step": 0,
        "start_epsilon": 1.0,
        "end_epsilon": 0.005,

        "update_pattern": (4, 1),
        "backprop_start_step": 10000,
        "replay_memory_size": 20000,
        "misc_scale": [0.01, 7 / 2100.0]
    }
    return QEngine(game=game, network_args=network_args, **engine_args)


def my_way_home():
    game = initialize_doom("config/my_way_home.cfg")
    network_args = {
        "ddqn": True,

    }
    engine_args = {
        "name": "my_way_home_dueling",
        "net_type": "dueling",
        "reshaped_x": 100,
        "reshaped_y": 75,
        "skiprate": 3,

        "remember_n_actions": 4,
        "count_states": True,
        "use_game_variables": True,
        # "shaping_on": True,

        "melt_steps": 10000,
        "epsilon_decay_steps": 500000,
        "epsilon_decay_start_step": 0,
        "start_epsilon": 1.0,
        "end_epsilon": 0.005,

        "update_pattern": (4, 1),
        "backprop_start_step": 10000,
        "replay_memory_size": 20000,
        # "misc_scale": [0.01, 7/2100.0]
    }
    return QEngine(game=game, network_args=network_args, **engine_args)


def take_cover():
    game = initialize_doom("config/take_cover.cfg")
    network_args = {
        "ddqn": True,
        "gamma": 1.0
    }
    engine_args = {
        "name": "take_covert",
        "net_type": "dueling",
        "reshaped_x": 100,
        "reshaped_y": 75,
        "skiprate": 3,

        "remember_n_actions": 4,
        "one_hot": True,
        "count_states": True,
        "use_game_variables": True,
        # "shaping_on": True,

        "melt_steps": 10000,
        "epsilon_decay_steps": 500000,
        "epsilon_decay_start_step": 0,
        "start_epsilon": 1.0,
        "end_epsilon": 0.005,

        "update_pattern": (4, 1),
        "backprop_start_step": 10000,
        "replay_memory_size": 20000,
        # "misc_scale": [0.01, 7 / 2100.0]
    }
    return QEngine(game=game, network_args=network_args, **engine_args)
