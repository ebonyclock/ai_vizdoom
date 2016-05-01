from lasagne.updates import rmsprop

from qengine import *


def initialize_doom(config_file):
    doom = DoomGame()
    doom.load_config("common.cfg")
    doom.load_config(config_file)

    print "Initializing DOOM ..."
    doom.init()
    print "DOOM initialized."
    return doom


def healthpicker(learning_rate=0.00001):
    cnn_args = dict()
    cnn_args["gamma"] = 1
    cnn_args["updates"] = rmsprop
    cnn_args["learning_rate"] = learning_rate

    architecture = dict()
    architecture["hidden_layers"] = 1
    architecture["hidden_units"] = [1024]
    architecture["conv_layers"] = 3
    architecture["pool_size"] = [(2, 2), (2, 2), (2, 2)]
    architecture["num_filters"] = [32, 32, 32]
    architecture["filter_size"] = [7, 5, 3]

    cnn_args["architecture"] = architecture
    name = "wumpus_alpha"
    return cnn_args, name


def setup_superhealth(net=healthpicker(), skiprate=10, learning_rate=0.00001, reshaped_x=120, mem=4):
    game = initialize_doom("superhealth.cfg")
    engine_args = dict()
    engine_args["game"] = game
    engine_args["network_args"] = net[0]
    engine_args["shaping_on"] = True
    engine_args["count_states"] = True
    engine_args["reward_scale"] = 0.01
    engine_args["misc_scale"] = [0.01, 1 / 2100.0]
    engine_args["epsilon_decay_steps"] = 100000
    engine_args["epsilon_decay_start_step"] = 4000

    engine_args['reshaped_x'] = reshaped_x
    engine_args["skiprate"] = skiprate
    engine_args["history_length"] = mem
    engine_args["batchsize"] = 64
    engine_args["name"] = net[1] + "shealth_" + "X" + str(reshaped_x) + "_skip" + str(skiprate) + "_lr" + str(
        learning_rate) + "_mem" + str(mem)
    return game, QEngine(**engine_args)


def default_basic():
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

    return cnn_args, "default"


def setup_basic(net=default_basic(), skiprate=4, learning_rate=0.01, reshaped_x=60):
    game = DoomGame()
    game.load_config("common.cfg")
    game.load_config("basic.cfg")

    print "Initializing DOOM ..."
    game.init()
    print "DOOM initialized."

    engine_args = dict()
    engine_args["evaluator"] = net[0]
    engine_args["game"] = game
    engine_args["reward_scale"] = 0.01
    engine_args['reshaped_x'] = 60
    engine_args["epsilon_decay_steps"] = 100000
    engine_args["epsilon_decay_start_step"] = 100000
    engine_args["batchsize"] = 40
    engine_args["name"] = net[1] + "shealth_" + "X" + reshaped_x + "_skip" + str(skiprate) + "_lr" + str(learning_rate)

    return game, QEngine(**engine_args)
