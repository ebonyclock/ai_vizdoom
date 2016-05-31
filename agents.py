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


def dqn_predict():
    game = initialize_doom("config/predict_position.cfg")
    engine_args = {
        "reshaped_x": 84,
        "reshaped_y": 84,
        "remember_n_actions": 4,
        "skiprate": 3,
        "name": "predict_dqn",
        "type": "dqn",
        "frozen_steps": 10000,
        "epsilon_decay_steps": 500000,
        "epsilon_decay_start_step": 0,
        "end_epsilon": 0.05,
        "start_epsilon": 1.0,
        "update_pattern": (4, 1),
        "backprop_start_step": 10000,
        "replay_memory_size": 50000
    }
    return game, QEngine(game=game, **engine_args)
