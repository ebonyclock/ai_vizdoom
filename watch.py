#!/usr/bin/python

import sys

from qengine import *

filename = "superhealth"
agent_loadfile = "params/superhealth/" + "vlad"
config_file = "superhealth" + ".cfg"

if len(sys.argv) > 1:
    agent_loadfile = sys.argv[1]
    if len(sys.argv) > 2:
        config_file = sys.argv[2]

game = DoomGame()
game.load_config("common.cfg")
game.load_config(config_file)

game.set_window_visible(True)
game.set_screen_resolution(ScreenResolution.RES_640X480)
# game.set_render_crosshair(True)

print "Initializing DOOM ..."
game.init()
print "\nDOOM initialized."

engine = QEngine.load(game, agent_loadfile)
engine.set_skiprate(6)
print "\nNetwork architecture:"
for p in get_all_param_values(engine.get_network()):
    print p.shape

episode_sleep = 0.5
action_sleep = 0.02

episodes = 20
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        engine.make_rendered_step(sleep_time=action_sleep)

        # s = game.get_state()
        # print "HP:",s.game_variables

    if episode_sleep > 0:
        sleep(episode_sleep)
    print i + 1, "Reward:", game.get_total_reward()
game.close()
