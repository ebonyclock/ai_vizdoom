#!/usr/bin/python

import sys
from time import time

from tqdm import tqdm

from agents import *
from util import *


setup = dqn_predict
training_steps_per_epoch = 200000
test_episodes_per_epoch = 300
save_params = False
save_results = False

epochs = np.inf
config_loadfile = None


results_loadfile = None
params_loadfile = None
load_params = False
load_results = False
if len(sys.argv) > 3:
    load_params = True
    load_results = True
    params_loadfile = sys.argv[1]
    results_loadfile = sys.argv[2]
    config_loadfile = sys.argv[3]
    params_savefile = params_loadfile
    results_savefile = results_loadfile

if load_params:
    game = initialize_doom(config_loadfile, True)
    engine = QEngine.load(game, params_loadfile)
else:
    game, engine = setup()
    basefile = engine.name
    params_savefile = "params/" + basefile
    results_savefile = "results/" + basefile + ".res"

results = None
epoch = 0
if load_results:
    results = pickle.load(open(results_loadfile, "r"))
    epoch = results["epoch"][-1] + 1
else:
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

print "\nNetwork architecture:"
for p in get_all_param_values(engine.get_network()):
    print p.shape
print "\nEngine setup:"
for k in engine.setup.keys():
    if k == "network_args":
        print"network_args:"
        net_args = engine.setup[k]
        for k2 in net_args.keys():
            print "\t",k2,":",net_args[k2]
    else:
        print k,":",engine.setup[k]
print
print "============================"

test_frequency = 1
overall_start = time()
if results_loadfile and len(results["time"]) > 0:
    overall_start -= results["overall_time"][-1]
# Training starts here!


while epoch < epochs:
    print "\nEpoch", epoch
    train_time = 0
    train_episodes_finished = 0
    mean_loss = 0
    if training_steps_per_epoch > 0:
        rewards = []

        start = time()
        engine.new_episode(update_state=True)
        print "\nTraining ..."
        for step in tqdm(range(training_steps_per_epoch)):
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
    if (epoch + 1) % test_frequency == 0 and test_episodes_per_epoch > 0:
        engine.learning_mode = False
        rewards = []

        start = time()
        print "Testing..."
        for test_episode in tqdm(range(test_episodes_per_epoch)):
            r = engine.run_episode()
            rewards.append(r)
        end = time()

        print "Test results:"
        print engine.get_actions_stats(clear=True, norm=False).reshape([-1, 4])
        rewards = np.array(rewards)
        print "mean:", rewards.mean(), "std:", rewards.std(), "max:", rewards.max(), "min:", rewards.min()
        print "t:", sec_to_str(end - start)

    overall_end = time()
    overall_time = overall_end - overall_start

    if save_results:
        print "Saving results to:", results_savefile
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
        res_f = open(results_savefile, 'w')
        pickle.dump(results, res_f)
        res_f.close()

    epoch += 1
    print ""

    if save_params:
        engine.save(params_savefile)

    print "Elapsed time:", sec_to_str(overall_time)
    print "========================="

overall_end = time()
print "Elapsed time:", sec_to_str(overall_end - overall_start)
