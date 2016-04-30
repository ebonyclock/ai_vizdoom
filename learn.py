#!/usr/bin/python

from time import time

from tqdm import tqdm
from vizdoom import *

from common import *
from qengine import *
import sys

filename = "superhealth"
config_file = filename + ".cfg"
setup = engine_setup_health


if len(sys.argv)>1:
    skiprate = int(sys.argv[1])
else:
    skiprate = 4
savefile = None
loadfile = None

suffix = "_lr0.0001"

savefile = "params/" + filename + "/skip" + str(skiprate) + suffix
# loadfile = "params/" + filename + "/skip" + str(skiprate) + suffix


results_savefile = "results/" + filename + "/skip" + str(skiprate) + suffix + ".res"
if results_savefile:
    results = dict()
    results["epoch"] = []
    results["time"] = []
    results["overall_time"] = []
    results["mean"] = []
    results["std"] = []
    results["max"] = []
    results["min"] = []
    results["epsilon"] = []
    results["traing_episodes_finished"] = []

game = DoomGame()
game.load_config("common.cfg")
game.load_config(config_file)
# game.set_doom_skill(0)

# game.clear_available_game_variables()
print "Initializing DOOM ..."
game.init()
print "DOOM initialized."

if loadfile:
    engine = QEngine.load(game, loadfile)
else:
    engine_args = setup(game)
    engine_args["skiprate"] = skiprate
    engine = QEngine(**engine_args)

print "\nNetwork architecture:"
for p in get_all_param_values(engine.get_network()):
    print p.shape

epochs = np.inf
training_steps_per_epoch = 2000
test_episodes_per_epoch = 200
test_frequency = 1
overall_start = time()

# Training starts here!
print
epoch = 0
while epoch < epochs:
    print "\nEpoch", epoch
    train_time = 0
    if training_steps_per_epoch > 0:
        rewards = []

        start = time()
        engine.new_episode(update_state=True)
        train_episodes_finished = 0
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
        print engine.get_actions_stats(True)

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
        print engine.get_actions_stats(clear=True, norm=False)
        rewards = np.array(rewards)
        print "mean:", rewards.mean(), "std:", rewards.std(), "max:", rewards.max(), "min:", rewards.min()
        print "t:", sec_to_str(end - start)

    overall_end = time()
    overall_time = overall_end - overall_start

    if results_savefile:
        print "Saving results to:", results_savefile
        results["epoch"].append(epoch)
        results["time"].append(train_time)
        results["overall_time"].append(overall_time)
        results["mean"].append(rewards.mean())
        results["std"].append(rewards.std())
        results["max"].append(rewards.max())
        results["min"].append(rewards.min())
        results["epsilon"].append(engine.get_epsilon())
        results["traing_episodes_finished"].append(train_episodes_finished)

        res_f = open(results_savefile, 'w')
        pickle.dump(results, res_f)
        res_f.close()

    epoch += 1
    print ""

    if savefile:
        engine.save(savefile)

    print "Elapsed time:", sec_to_str(overall_time)
    print "========================="

overall_end = time()
print "Elapsed time:", sec_to_str(overall_end - overall_start)
