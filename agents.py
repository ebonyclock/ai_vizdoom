from lasagne.updates import rmsprop

from qengine import *


def initialize_doom(config_file, grayscale=False):
    doom = DoomGame()
    doom.load_config("common.cfg")
    doom.load_config(config_file)
    if grayscale:
        doom.set_screen_format(ScreenFormat.GRAY8)
    print "Initializing DOOM ..."
    doom.init()
    print "DOOM initialized."
    return doom


def setup_basic(skiprate=4):
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

    game = initialize_doom("basic.cfg")


    engine_args = dict()
    engine_args["network_args"] = cnn_args
    engine_args["game"] = game
    engine_args["reward_scale"] = 0.01
    engine_args['reshaped_x'] = 60
    engine_args["epsilon_decay_steps"] = 100000
    engine_args["epsilon_decay_start_step"] = 100000
    engine_args["batchsize"] = 40
    engine_args["name"] = "original_basic_skip" + str(skiprate)
    engine_args["skiprate"] = skiprate
    
    return game, QEngine(**engine_args)


def superhealth_engine_base(grayscale=False):
    game = initialize_doom("superhealth.cfg", grayscale)
    engine_args = dict()
    engine_args["game"] = game
    engine_args["reward_scale"] = 0.01
    engine_args["misc_scale"] = [0.01, 1 / 2100.0]
    engine_args["batchsize"] = 64
    engine_args["count_states"] = True
    return engine_args


def setup_vlad():
    cnn_args = dict()
    cnn_args["gamma"] = 1
    cnn_args["updates"] = rmsprop
    cnn_args["learning_rate"] = 0.00005

    # Architecture
    architecture = dict()
    architecture["hidden_layers"] = 1
    architecture["hidden_units"] = [1024]
    architecture["conv_layers"] = 3
    architecture["pool_size"] = [(2, 2), (2, 2), (2, 2)]
    architecture["num_filters"] = [32, 32, 32]
    architecture["filter_size"] = [7, 5, 3]
    cnn_args["architecture"] = architecture

    engine_args = superhealth_engine_base()
    engine_args["network_args"] = cnn_args
    engine_args["shaping_on"] = True

    engine_args["start_epsilon"] = 0.9
    engine_args["end_epsilon"] = 0.005
    engine_args["epsilon_decay_steps"] = 100000
    engine_args["epsilon_decay_start_step"] = 10000

    engine_args['reshaped_x'] = 120
    engine_args["skiprate"] = 10
    engine_args["history_length"] = 4

    engine_args["name"] = "Vlad"

    return engine_args["game"], QEngine(**engine_args)


def setup_vlad_memorytest():
    cnn_args = dict()
    cnn_args["gamma"] = 1
    cnn_args["updates"] = rmsprop
    cnn_args["learning_rate"] = 0.00001

    # Architecture
    architecture = dict()
    architecture["hidden_layers"] = 2
    architecture["hidden_units"] = [1024, 1024]
    architecture["conv_layers"] = 3
    architecture["pool_size"] = [(2, 2), (2, 2), (2, 2)]
    architecture["num_filters"] = [32, 32, 32]
    architecture["filter_size"] = [7, 5, 3]
    architecture["memory"] = 4
    cnn_args["architecture"] = architecture

    engine_args = superhealth_engine_base()
    engine_args["network_args"] = cnn_args
    engine_args["shaping_on"] = True

    engine_args["start_epsilon"] = 0.9
    engine_args["end_epsilon"] = 0.005
    engine_args["epsilon_decay_steps"] = 100000
    engine_args["epsilon_decay_start_step"] = 10000

    engine_args["reshaped_x"] = 120
    engine_args["skiprate"] = 10
    engine_args["history_length"] = 4
    engine_args["type"] = "cnn_mem"

    engine_args["name"] = "eVlad"

    return engine_args["game"], QEngine(**engine_args)

def setup_vlad_grayscale():
    cnn_args = dict()
    cnn_args["gamma"] = 1
    cnn_args["updates"] = rmsprop
    cnn_args["learning_rate"] = 0.00001

    # Architecture
    architecture = dict()
    architecture["hidden_layers"] = 2
    architecture["hidden_units"] = [1024, 1024]
    architecture["conv_layers"] = 3
    architecture["pool_size"] = [(2, 2), (2, 2), (2, 2)]
    architecture["num_filters"] = [32, 32, 32]
    architecture["filter_size"] = [7,5,3]
    architecture["memory"] = 4
    cnn_args["architecture"] = architecture

    engine_args = superhealth_engine_base(grayscale=True)
    engine_args["network_args"] = cnn_args
    engine_args["shaping_on"] = True

    engine_args["start_epsilon"] = 0.9
    engine_args["end_epsilon"] = 0.005
    engine_args["epsilon_decay_steps"] = 100000
    engine_args["epsilon_decay_start_step"] = 10000

    engine_args["reshaped_x"] = 120
    engine_args["skiprate"] = 10
    engine_args["history_length"] = 4
    engine_args["type"] = "cnn_mem"

    engine_args["name"] = "eVlad"

    return engine_args["game"], QEngine(**engine_args)
