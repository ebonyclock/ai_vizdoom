#!/usr/bin/python

import pickle
from time import time
import numpy as np
import agents
from args_parser import build_learn_parser
from qengine import QEngine
from util import *
from inspect import getmembers, isfunction


params_parser = build_learn_parser()
args = params_parser.parse_args()


if args.list_agents:
    print "Available agents in agents.py:"
    for member in getmembers(agents):
        if isfunction(member[1]):
            if member[1].__name__[0] !="_":
                print "  ",member[1].__name__
    exit(0)

if args.agent is not None:
    setup = getattr(agents, args.agent)

training_steps_per_epoch = args.train_steps
epochs = args.epochs
test_episodes_per_epoch = args.test_episodes
save_params = not args.no_save
save_results = not args.no_save_results
config_loadfile = args.config_file
best_result_so_far = None
save_best = not args.no_save_best
if args.no_tqdm:
    my_range = xrange
else:
    from tqdm import trange
    my_range = trange

agent_loadfile = args.agent_file


results = None
if agent_loadfile:
    engine = QEngine.load(agent_loadfile, config_file=config_loadfile)
    results = pickle.load(open(engine.results_file), "r")
else:
    if args.name is not None:
        engine = setup(args.name)
    else:
        engine = setup()
    game = engine.game

    if save_results:
        results = dict()
        results["epoch"] = []
        results["time"] = []
        results["overall_time"] = []
        results["mean"] = []
        results["std"] = []
        results["max"] = []
        results["min"] = []
        results["epsilon"] = []
        results["training_episodes_finished"] = []
        results["loss"] = []
        results["setup"] = engine.setup
        results["best"] = None
        results["actions"] = []


engine.print_setup()
print "\n============================"

epoch = 1
overall_start = time()
if save_results and len(results["epoch"]) > 0:
    overall_start -= results["overall_time"][-1]
    epoch = results["epoch"][-1] + 1
    best_result_so_far = results["best"]
    if "actions" not in results:
        results["actions"] = []
        for _ in len(results["epoch"]):
            results["actions"].append(0)


while epoch - 1 < epochs:
    print "\nEpoch", epoch
    train_time = 0
    train_episodes_finished = 0
    mean_loss = 0
    if training_steps_per_epoch > 0:
        rewards = []

        start = time()
        engine.new_episode(update_state=True)
        print "\nTraining ..."
        for step in my_range(training_steps_per_epoch):
            if game.is_episode_finished():
                r = game.get_total_reward()
                rewards.append(r)
                engine.new_episode(update_state=True)
                train_episodes_finished += 1
            engine.make_learning_step()
        end = time()
        train_time = end - start

        print train_episodes_finished, "training episodes played."
        print "Training results:"
        print engine.get_actions_stats(clear=True).reshape([-1, 4])

        mean_loss = engine._evaluator.get_mean_loss()

        if len(rewards) == 0:
            rewards.append(-123)
        rewards = np.array(rewards)

        print "mean:", rewards.mean(), "std:", rewards.std(), "max:", rewards.max(), "min:", rewards.min(), "mean_loss:", mean_loss, "eps:", engine.get_epsilon()
        print "t:", sec_to_str(train_time)

    # learning mode off
    if  test_episodes_per_epoch > 0:
        engine.learning_mode = False
        rewards = []

        start = time()
        print "Testing..."
        for test_episode in my_range(test_episodes_per_epoch):
            r = engine.run_episode()
            rewards.append(r)
        end = time()

        print "Test results:"
        print engine.get_actions_stats(clear=True, norm=False).reshape([-1, 4])
        rewards = np.array(rewards)
        best_result_so_far = max(best_result_so_far, rewards.mean())
        print "mean:", rewards.mean(), "std:", rewards.std(), "max:", rewards.max(), "min:", rewards.min()
        print "t:", sec_to_str(end - start)
        print "Best so far:", best_result_so_far

    overall_end = time()
    overall_time = overall_end - overall_start

    if save_results:
        print "Saving results to:", engine.results_file
        results["epoch"].append(epoch)
        results["time"].append(train_time)
        results["overall_time"].append(overall_time)
        results["mean"].append(rewards.mean())
        results["std"].append(rewards.std())
        results["max"].append(rewards.max())
        results["min"].append(rewards.min())
        results["epsilon"].append(engine.get_epsilon())
        results["training_episodes_finished"].append(train_episodes_finished)
        results["loss"].append(mean_loss)
        results["best"] = best_result_so_far
        results["actions"].append(engine.steps)
        res_f = open(engine.results_file, 'w')
        pickle.dump(results, res_f)
        res_f.close()

    epoch += 1
    print ""

    if save_params:
        engine.save()
    if save_best:
        engine.save(engine.params_file+"_best")

    print "Elapsed time:", sec_to_str(overall_time)
    print "========================="

overall_end = time()
print "Elapsed time:", sec_to_str(overall_end - overall_start)
