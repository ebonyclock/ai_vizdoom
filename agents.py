from lasagne.updates import rmsprop

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


def superhealth_setup(actions=0,grayscale=True):
    game = initialize_doom("config/health_gathering_supreme.cfg", grayscale)
    engine_args = dict()
    engine_args["game"] = game
    engine_args["reward_scale"] = 0.01
    engine_args["misc_scale"] = [0.01, 1 / 2100.0]
    engine_args["batchsize"] = 64
    engine_args["count_states"] = True
    engine_args["shaping_on"] = True
    engine_args["end_epsilon"] = 0.05
    engine_args["reshaped_x"] = 120
    engine_args["remember_n_actions"] = actions
    engine_args["skiprate"] = 10
    engine_args["name"]="superhealth"

    return engine_args["game"], QEngine(**engine_args)

def cover_setup(actions=0,grayscale=True):
    game = initialize_doom("config/take_cover.cfg", grayscale)
    engine_args = dict()
    engine_args["game"] = game
    engine_args["reward_scale"] = 0.01
    engine_args["misc_scale"] = [0.01, 1 / 2100.0]
    engine_args["batchsize"] = 64
    engine_args["count_states"] = True
    engine_args["end_epsilon"] = 0.05
    engine_args["reshaped_x"] = 120
    engine_args["remember_n_actions"] = actions
    engine_args["skiprate"] = 4
    engine_args["name"]="cover"
    return engine_args["game"], QEngine(**engine_args)

def predict_setup(actions=0,grayscale=True):
    game = initialize_doom("config/predict_position.cfg", grayscale)
    engine_args = dict()
    engine_args["game"] = game
    engine_args["batchsize"] = 64
    engine_args["end_epsilon"] = 0.05
    engine_args["reshaped_x"] = 120
    engine_args["remember_n_actions"] = actions
    engine_args["skiprate"] = 4
    engine_args["name"]="predict"
    return engine_args["game"], QEngine(**engine_args)

def line_setup(actions=0,grayscale=True):
    game = initialize_doom("config/defend_the_line.cfg", grayscale)
    engine_args = dict()
    engine_args["game"] = game
    engine_args["batchsize"] = 64
    engine_args["count_states"] = True
    engine_args["misc_scale"] = [0.01, 1 / 2100.0]
    engine_args["end_epsilon"] = 0.05
    engine_args["reshaped_x"] = 120
    engine_args["remember_n_actions"] = actions
    engine_args["skiprate"] = 4
    engine_args["name"]="line"
    return engine_args["game"], QEngine(**engine_args)



def line_center(actions=0,grayscale=True):
    game = initialize_doom("config/defend_the_center.cfg", grayscale)
    engine_args = dict()
    engine_args["game"] = game
    engine_args["batchsize"] = 64
    engine_args["count_states"] = True
    engine_args["misc_scale"] = [1/26.0, 0.01, 1 / 2100.0]
    engine_args["end_epsilon"] = 0.05
    engine_args["reshaped_x"] = 120
    engine_args["remember_n_actions"] = actions
    engine_args["skiprate"] = 4
    engine_args["name"]="center"
    return engine_args["game"], QEngine(**engine_args)

def dqn_predict(actions=0,grayscale=True):
    game = initialize_doom("config/predict_position.cfg", grayscale)
    engine_args = dict()
    engine_args["game"] = game
    engine_args["batchsize"] = 64
    engine_args["end_epsilon"] = 0.05
    engine_args["reshaped_x"] = 84
    engine_args["reshaped_y"] = 84
    engine_args["remember_n_actions"] = actions
    engine_args["skiprate"] = 4
    engine_args["name"]="predict_dqn"
    engine_args["type"] = "dqn"
    return engine_args["game"], QEngine(**engine_args)



