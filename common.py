from lasagne.updates import rmsprop
from vizdoom import *

from evaluators import *


# Changes seconds to some nice string
def sec_to_str(sec):
    res = str(int(sec % 60)) + "s"
    sec = int(sec / 60)
    if sec > 0:
        res = str(int(sec % 60)) + "m " + res
        sec = int(sec / 60)
        if sec > 0:
            res = str(int(sec % 60)) + "h " + res
            sec = int(sec / 60)
    return res


def left_right_move_actions():
    idle = [0, 0, 0]
    left = [1, 0, 0]
    right = [0, 1, 0]
    move = [0, 0, 1]
    move_left = [1, 0, 1]
    move_right = [0, 1, 1]
    return [idle, left, right, move]


def engine_setup_basic(game):
    cnn_args = dict()
    cnn_args["gamma"] = 0.99
    cnn_args["updates"] = sgd
    cnn_args["learning_rate"] = 0.01

    architecture = dict(hidden_units=[800], hidden_layers=1)
    architecture["conv_layers"] = 2
    architecture["pool_size"] = [(2, 2), (2, 2)]
    architecture["num_filters"] = [32, 32]
    architecture["filter_size"] = [7, 4]
    architecture["output_nonlin"] = None

    cnn_args["architecture"] = architecture
    engine_args = dict()
    engine_args["evaluator"] = cnn_args
    engine_args["game"] = game
    engine_args["reward_scale"] = 0.01
    engine_args['reshaped_x'] = 60
    engine_args["epsilon_decay_steps"] = 100000
    engine_args["epsilon_decay_start_step"] = 100000
    engine_args["batchsize"] = 40

    return engine_args


def engine_setup_health(game):
    cnn_args = dict()
    cnn_args["gamma"] = 0.99
    cnn_args["updates"] = rmsprop
    cnn_args["learning_rate"] = 0.0001
    # cnn_args["max_q"] = 21

    architecture = dict(hidden_units=[1024], hidden_layers=1)
    architecture["conv_layers"] = 3
    architecture["pool_size"] = [(2, 2), (2, 2), (2, 2)]
    architecture["num_filters"] = [32, 32, 32]
    architecture["filter_size"] = [7, 5, 3]
    # network_args["dropout"] = False
    # network_args["output_nonlin"] = None
    # network_args["output_nonlin"] = create_cutoff(2100)
    # network_args["hidden_nonlin"] = None

    cnn_args["architecture"] = architecture

    engine_args = dict()
    engine_args["network_args"] = cnn_args
    engine_args["game"] = game
    engine_args["reward_scale"] = 0.01

    engine_args['reshaped_x'] = 120
    engine_args["shaping_on"] = True
    engine_args["count_states"] = True
    engine_args["misc_scale"] = [0.01, 1 / 2100.0]
    engine_args["epsilon_decay_steps"] = 100000
    engine_args["epsilon_decay_start_step"] = 4000
    engine_args["batchsize"] = 64
    # engine_args["history_length"] = 4
    return engine_args
