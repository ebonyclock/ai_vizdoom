#!/usr/bin/python

import argparse
from time import sleep

import numpy as np

from qengine import QEngine

parser = argparse.ArgumentParser(description='A script to watch agents play or test them.')

parser.add_argument('agent_file', metavar='agent_file', type=str, nargs="?",
                    default=None,
                    help='file with the agent')

parser.add_argument('--config-file', '-c', metavar='config_file', type=str, nargs='?', default=None,
                    help='override agent\'s configuration file')

parser.add_argument('--episodes', "-e", metavar='episodes', type=int, nargs='?', default=20,
                    help='run this many episodes (default 20)')

parser.add_argument('--no-watch', dest='no_watch', action='store_const',
                    const=True, default=False,
                    help='do not display the window and do not sleep')
parser.add_argument('--action-sleep', "-s", metavar='action_sleep', type=float, nargs='?', default=1 / 35.0,
                    help='sleep this many seconds after each action (default=1/35.0)')
parser.add_argument('--episode-sleep', metavar='episode_sleep', type=float, nargs='?', default=0.5,
                    help='sleep this many seconds after each episode (default=0.5)')

args = parser.parse_args()

engine = QEngine.load(args.agent_file, config_file=args.config_file)

if args.no_watch:
    episode_sleep = 0
    action_sleep = 0
else:
    episode_sleep = args.episode_sleep
    action_sleep = args.action_sleep
    engine.game.close()
    engine.game.set_window_visible(True)
    engine.game.init()

engine.print_setup()

episodes = args.episodes
rewards = []
for i in range(episodes):
    r = engine.run_episode(sleep_time=action_sleep)
    rewards.append(r)
    print i + 1, "Reward:", r
    if episode_sleep > 0:
        sleep(episode_sleep)

print "Mean rewards:", np.mean(rewards)
