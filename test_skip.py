#!/usr/bin/python

from tqdm import tqdm
from vizdoom import *

from qengine import *

files_base = "params/basic/basic_skip"
skips = [0,1,2,3, 4, 5, 6, 7, 10, 15, 20, 25, 30, 35, 40]

game = DoomGame()
game.load_config("common.cfg")
game.load_config("basic.cfg")
game.init()

episodes = 10000
print "RESUTLS FROM", episodes, "EPISODES:"
print "train skip | mean | std | min | max"
for skip in skips:
    engine = QEngine.load(game, files_base + str(skip), quiet=True)
    train_skiprate = engine.get_skiprate()
    engine.set_skiprate(10)
    engine.learning_mode = False
    rewards = []

    for test_episode in tqdm(range(episodes)):
        r = engine.run_episode()
        rewards.append(r)

    print train_skiprate, np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)
