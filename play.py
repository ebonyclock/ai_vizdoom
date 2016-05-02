#!/usr/bin/python
from vizdoom import *

game = DoomGame()
game.load_config("common.cfg")
game.load_config("superhealth.cfg")
game.set_mode(Mode.SPECTATOR)
game.set_window_visible(True)
# game.set_render_hud(True)
game.set_screen_resolution(ScreenResolution.RES_640X480)

game.init()

skiprate = 0
episodes = 20
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        game.advance_action(skiprate + 1)
        print doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
        print game.get_last_action()
    print i + 1, "Reward:", game.get_total_reward()
game.close()
