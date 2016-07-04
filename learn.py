#!/usr/bin/python

import pickle
from inspect import getmembers, isfunction
from time import time
import numpy as np
import agents
from args_parser import build_learn_parser
from qengine import QEngine
from util import *
import yaml

# Arguments parsing
params_parser = build_learn_parser()
args = params_parser.parse_args()
training_steps_per_epoch = args.train_steps[0]
epochs = args.epochs[0]
test_episodes_per_epoch = args.test_episodes[0]
save_params = not args.no_save
save_results = not args.no_save_results
best_result_so_far = None
save_best = not args.no_save_best

if args.no_tqdm:
    my_range = xrange
else:
    from tqdm import trange

    my_range = trange

if args.list_agents:
    print "Available agents in agents.py:"
    for member in getmembers(agents):
        if isfunction(member[1]):
            if member[1].__name__[0] != "_":
                print "  ", member[1].__name__
    exit(0)


results = None
if args.agent_file[0]:
    engine = QEngine.load(args.agent_file[0], config_file=args.config_fil1[0])
    results = pickle.load(open(engine.results_file, "r"))
else:
    if args.agent is not None:
        if args.name[0] is not None:
            engine = getattr(agents, args.agent)(args.name[0])
        else:
            engine = getattr(agents, args.agent)()

    elif args.json_file[0] is not None:
        engine_args = yaml.load(open(args.json_file[0], "r"))
        if args.name[0] is not None:
            engine_args["name"] = args.name[0]
        if args.config_file[0] is not None:
            engine_args["config_file"] = args.config_file[0]
        engine = QEngine(**engine_args)
    else:
        raise Exception("No agent file, no json, no agent ... Huge fuckup. Aborting.")

    if save_results:
        results = dict()
        results["epoch"] = []
        results["time"] = []
        results["overall_time"] = []
        results["mean"] = []
        results["std"] = []
        results["train_mean"] = []
        results["train_std"] = []
        results["max"] = []
        results["min"] = []
        results["epsilon"] = []
        results["training_episodes_finished"] = []
        results["loss"] = []
        results["setup"] = engine.setup
        results["best"] = None
        results["actions"] = []
# Arguments parsing ends here

game = engine.game
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
        train_rewards = []

        start = time()
        engine.new_episode(update_state=True)
        print "\nTraining ..."
        for step in my_range(training_steps_per_epoch):
            #TODO move the check to QEngine
            if game.is_episode_finished():
                r = game.get_total_reward()
                train_rewards.append(r)
                engine.new_episode(update_state=True)
                train_episodes_finished += 1
            engine.make_learning_step()
        end = time()
        train_time = end - start

        print train_episodes_finished, "training episodes played."
        print "Training results:"
        print engine.get_actions_stats(clear=True).reshape([-1, 4])

        mean_loss = engine.get_mean_loss()

        if len(train_rewards) == 0:
            train_rewards.append(-123)
        train_rewards = np.array(train_rewards)

        print "mean:", train_rewards.mean(), "std:", train_rewards.std(), "max:", train_rewards.max(), "min:", train_rewards.min(), "mean_loss:", mean_loss, "eps:", engine.get_epsilon()
        print "t:", sec_to_str(train_time)

    # learning mode off
    new_best = False
    if test_episodes_per_epoch > 0:
        engine.learning_mode = False
        test_rewards = []

        start = time()
        print "Testing..."
        for test_episode in my_range(test_episodes_per_epoch):
            r = engine.run_episode()
            test_rewards.append(r)
        end = time()

        print "Test results:"
        print engine.get_actions_stats(clear=True, norm=False).reshape([-1, 4])
        test_rewards = np.array(test_rewards)

        if test_rewards.mean() >= best_result_so_far:
            best_result_so_far = test_rewards.mean()
            new_best = True
        else:
            new_best = False
        print "mean:", test_rewards.mean(), "std:", test_rewards.std(), "max:", test_rewards.max(), "min:", test_rewards.min()
        print "t:", sec_to_str(end - start)
        print "Best so far:", best_result_so_far

    overall_end = time()
    overall_time = overall_end - overall_start

    if save_results:
        print "Saving results to:", engine.results_file
        results["epoch"].append(epoch)
        results["time"].append(train_time)
        results["overall_time"].append(overall_time)
        results["mean"].append(test_rewards.mean())
        results["std"].append(test_rewards.std())
        results["max"].append(test_rewards.max())
        results["min"].append(test_rewards.min())
        results["train_mean"].append(train_rewards.mean())
        results["train_std"].append(train_rewards.std())

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
    if save_best and new_best:
        engine.save(engine.params_file + "_best")

    print "Elapsed time:", sec_to_str(overall_time)
    print "========================="

overall_end = time()
print "Elapsed time:", sec_to_str(overall_end - overall_start)
