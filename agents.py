import numpy as np
from qengine import QEngine


def _merge_dicts(dicta, dictb):
    ret = dict()
    for key in dicta.keys():
        ret[key] = dicta[key]

    for key in dictb.keys():
        ret[key] = dictb[key]
    return ret


def _default_engine_args():
    default_network_args = {
        "ddqn": True,
        "learning_rate": 0.00025,
        "gamma": 1.0
    }

    default_args = {
        "net_type": "dueling",
        "reshaped_x": 100,
        "reshaped_y": 75,
        "skiprate": 3,

        "history_length": 4,
        "remember_n_actions": 4,
        "one_hot_nactions": True,

        "use_game_variables": True,
        "count_time": True,
        "shaping_on": False,

        "melt_steps": 10000,
        "epsilon_decay_steps": 500000,
        "epsilon_decay_start_step": 0,
        "start_epsilon": 1.0,
        "end_epsilon": 0.005,

        "update_pattern": (4, 1),
        "backprop_start_step": 10000,
        "replay_memory_size": 20000,
        "misc_scale": None,
        "reward_scale": None,
        "batchsize": 64

    }
    return default_args, default_network_args


def predict(name="predict_def"):
    defaults, net_defaults = _default_engine_args()
    custom_args = {
        "name": name,
        "config_file": "config/predict_position.cfg"
    }
    network_args = net_defaults
    engine_args = _merge_dicts(defaults, custom_args)
    return QEngine(network_args=network_args, **engine_args)


def predict_supreme(name="predict-s_def"):
    defaults, net_defaults = _default_engine_args()
    custom_args = {
        "name": name,
        "config_file": "config/predict_position_supreme.cfg"
    }
    network_args = net_defaults
    engine_args = _merge_dicts(defaults, custom_args)
    return QEngine(network_args=network_args, **engine_args)


def health_supreme(name="health-s_def_skip7"):
    defaults, net_defaults = _default_engine_args()
    custom_args = {
        "name": name,
        "config_file": "config/health_gathering_supreme.cfg",
        "skiprate": 7,
    }
    network_args = net_defaults
    engine_args = _merge_dicts(defaults, custom_args)
    return QEngine(network_args=network_args, **engine_args)


def health(name="health_def_skip7"):
    defaults, net_defaults = _default_engine_args()
    custom_args = {
        "name": name,
        "config_file": "config/health_gathering.cfg",
        "skiprate": 7,
    }
    network_args = net_defaults
    engine_args = _merge_dicts(defaults, custom_args)
    return QEngine(network_args=network_args, **engine_args)


def defend_the_center(name="center_def"):
    defaults, net_defaults = _default_engine_args()
    custom_args = {
        "name": name,
        "config_file": "config/defend_the_center.cfg"
    }
    network_args = net_defaults
    engine_args = _merge_dicts(defaults, custom_args)
    return QEngine(network_args=network_args, **engine_args)


def take_cover(name="cover_def"):
    defaults, net_defaults = _default_engine_args()
    custom_args = {
        "name": name,
        "config_file": "config/take_cover.cfg"
    }
    network_args = net_defaults
    engine_args = _merge_dicts(defaults, custom_args)
    return QEngine(network_args=network_args, **engine_args)


def take_cover_simple(name="cover_simple_def"):
    defaults, net_defaults = _default_engine_args()
    custom_args = {
        "name": name,
        "config_file": "config/take_cover_simple.cfg"
    }
    network_args = net_defaults
    engine_args = _merge_dicts(defaults, custom_args)
    return QEngine(network_args=network_args, **engine_args)


def pacman(name="pacman_def"):
    defaults, net_defaults = _default_engine_args()
    custom_args = {
        "name": name,
        "config_file": "config/pacman.cfg"
    }
    network_args = net_defaults
    engine_args = _merge_dicts(defaults, custom_args)
    return QEngine(network_args=network_args, **engine_args)


def health_baseline(name="health_baseline4"):
    defaults, net_defaults = _default_engine_args()
    custom_args = {
        "name": name,
        "config_file": "config/health_gathering.cfg",
        "skiprate": 9,
        "count_time": False,
        "reshaped_x": 84,
        "reshaped_y": 84,
        "net_type": "dqn",
    }
    network_args = net_defaults
    net_defaults["gamma"] = 0.99
    engine_args = _merge_dicts(defaults, custom_args)
    return QEngine(network_args=network_args, **engine_args)
