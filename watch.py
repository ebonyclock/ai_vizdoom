#!/usr/bin/python
from common import *
from vizdoom import ScreenResolution

filename = "superhealth"
loadfile = "params/"+filename+"_skip4"
config_file = "superhealth" + ".cfg"

game = DoomGame()
game.load_config("common.cfg")
game.load_config(config_file)

game.set_window_visible(True)
#game.set_render_crosshair(True)


print "Initializing DOOM ..."
game.init()
print "\nDOOM initialized."

engine = QEngine.load(game, loadfile)
print "\nNetwork architecture:"
for p in get_all_param_values(engine.get_network()):
	print p.shape


episode_sleep = 0.5

episodes = 20
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        engine.make_rendered_step(sleep_time = 0.02)

        #s = game.get_state()
        #print "HP:",s.game_variables
    
    if episode_sleep>0:
        sleep(episode_sleep)
    print i+1,"Reward:", game.get_summary_reward()
game.close()
