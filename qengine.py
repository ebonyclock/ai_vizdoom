import itertools as it
import pickle
import random
from random import choice
from time import sleep

import cv2
import numpy as np
from lasagne.layers import get_all_param_values
from lasagne.layers import set_all_param_values
from vizdoom import *
from evaluators import *

from transition_bank import TransitionBank


def default_actions_generator(the_game):
    n = the_game.get_available_buttons_size()
    actions = []
    for perm in it.product([0, 1], repeat=n):
        actions.append(list(perm))
    return actions


# TODO get rid af actions_generator
# TODO to something with loading and saving the network
class QEngine:
    def __init__(self, **kwargs):
        self._qengine_args = kwargs
        self._initialize(**kwargs)
        kwargs["game"] = None

    def _prepare_for_save(self):
        self._qengine_args["epsilon"] = self._epsilon
        self._qengine_args["steps"] = self._steps
        self._qengine_args["skiprate"] = self._skiprate

    def _initialize(self, game,  network_args, history_length=1, batchsize=64,
                    update_pattern=(4, 4),
                    bank_capacity=10000, start_epsilon=1.0, end_epsilon=0.1, epsilon_decay_start_step=100000,
                    epsilon_decay_steps=100000,
                    reward_scale=1.0, misc_scale=None, max_reward=None, reshaped_x=120, skiprate=0,
                    shaping_on=False, count_states=False, actions=None):
        # Line that makes sublime collapse code correctly


        if count_states is not None:
            self._count_states = bool(count_states)
        self._max_reward = np.float32(max_reward)
        self._reward_scale = reward_scale
        self._game = game
        self._batchsize = batchsize
        self._history_length = max(history_length, 1)
        self._update_pattern = update_pattern
        self._epsilon = max(min(start_epsilon, 1.0), 0.0)
        self._end_epsilon = min(max(end_epsilon, 0.0), self._epsilon)
        self._epsilon_decay_steps = epsilon_decay_steps
        self._epsilon_decay_stride = (self._epsilon - end_epsilon) / epsilon_decay_steps
        self._epsilon_decay_start = epsilon_decay_start_step
        self._skiprate = max(skiprate, 0)
        self._shaping_on = shaping_on
        self._steps = 0
        if self._shaping_on:
            self._last_shaping_reward = 0

        self.learning_mode = True

        if actions == None:
            self._actions = default_actions_generator(game)
        else:
            self._actions = actions
        self._actions_num = len(self._actions)
        self._actions_stats = np.zeros([self._actions_num], np.int)

        # change img_shape according to the history size
        self._channels = game.get_screen_channels()
        if self._history_length > 1:
            self._channels *= self._history_length

        self._scale = float(reshaped_x) / game.get_screen_width()
        y = int(game.get_screen_height()*self._scale)
        x = reshaped_x


        img_shape = [self._channels, y, x]

        if self._scale == 1:

            def convert(img):
                img = np.float32(img) / 255.0
                return img
        else:
            def convert(img):
                img = np.float32(img) / 255.0
                new_image = np.ndarray([img.shape[0], y, x], dtype=np.float32)
                for i in range(img.shape[0]):
                    new_image[i] = cv2.resize(img[i], (x, y))
                return new_image

        self._convert_image = convert

        self._misc_len = game.get_available_game_variables_size() + self._count_states
        if self._misc_len > 0:
            self._misc_state_included = True
            self._current_misc_state = np.zeros(self._misc_len * self._history_length, dtype=np.float32)
            self._misc_buffer = np.zeros(self._misc_len, dtype=np.float32)
            if misc_scale is not None:
                self._misc_scale = np.array(misc_scale, dtype=np.float32)
            else:
                self._misc_scale = None
        else:
            self._misc_state_included = False

        state_format = dict()
        state_format["s_img"] = img_shape
        state_format["s_misc"] = self._misc_len * self._history_length
        self._transitions = TransitionBank(state_format, bank_capacity, batchsize)

        if "type" not in network_args.keys():
            eval_type=None
        else:
            eval_type = network_args["type"]
            del(network_args["type"])
        network_args["state_format"] = state_format
        network_args["actions_number"] = len(self._actions)
        if eval_type in ("cnn",None,""):
            self._evaluator = CNNEvaluator(**network_args)
        else:
            print "Unsupported evaluator type specified"

        self._current_image_state = np.zeros(img_shape, dtype=np.float32)

    def _update_state(self):
        raw_state = self._game.get_state()
        img = self._convert_image(raw_state.image_buffer)

        if self._misc_state_included:
            misc_len = self._misc_len
            misc = self._misc_buffer
            misc[0:misc_len - self._count_states] = np.float32(raw_state.game_variables)
            misc[-1] = raw_state.number
            if self._misc_scale is not None:
                misc = misc * self._misc_scale

        if self._history_length > 1:
            pure_channels = self._channels / self._history_length
            self._current_image_state[0:-pure_channels] = self._current_image_state[pure_channels:]
            self._current_image_state[-pure_channels:] = img

            if self._misc_state_included:
                border = (self._history_length - 1) * self._misc_len
                self._current_misc_state[0:border] = self._current_misc_state[self._misc_len:]
                self._current_misc_state[border:] = misc

        else:
            self._current_image_state[:] = img
            if self._misc_state_included:
                self._current_misc_state[:] = misc

    def new_episode(self, update_state=False):
        self._game.new_episode()
        self.reset_state()
        self._last_shaping_reward = 0
        if update_state:
            self._update_state()

    # Return current state including history
    def _current_state(self):
        if self._misc_state_included:
            s = [self._current_image_state, self._current_misc_state]
        else:
            s = [self._current_image_state]
        return s

    # Return current state's COPY including history.
    def _current_state_copy(self):
        if self._misc_state_included:
            s = [self._current_image_state.copy(), self._current_misc_state.copy()]
        else:
            s = [self._current_image_state.copy()]
        return s

    # Sets the whole state to zeros. 
    def reset_state(self):
        self._current_image_state.fill(0.0)
        if self._misc_state_included:
            self._current_misc_state.fill(0.0)

    def make_step(self):
        self._update_state()
        # current_state_COPY - copy is here cause tests go worse than training
        a = self._evaluator.best_action(self._current_state_copy())
        self._actions_stats[a] += 1
        self._game.make_action(self._actions[a], self._skiprate + 1)

    def make_rendered_step(self, sleep_time=0):
        self._update_state()
        a = self._evaluator.best_action(self._current_state_copy())
        self._actions_stats[a] += 1

        self._game.set_action(self._actions[a])
        for i in range(self._skiprate):
            self._game.advance_action(1, False, True)
            sleep(sleep_time)
        self._game.advance_action()
        sleep(sleep_time)

    # Updates the state and returns the best action.
    def best_action(self, state):
        self._update_state()
        return self._actions[self._evaluator.best_action(self._current_state())]

    # Makes a random action without state update.
    def make_random_step(self):
        self._game.make_action(choice(self._actions), self._skiprate + 1)

    # Performs a learning step according to epsilon-greedy policy.
    # The step spans self._skiprate +1 actions.
    def make_learning_step(self):
        self._steps += 1
        # epsilon decay
        if self._steps > self._epsilon_decay_start and self._epsilon > self._end_epsilon:
            self._epsilon -= self._epsilon_decay_stride
            self._epsilon = max(self._epsilon, 0)

            # Copy because state will be changed in a second
        s = self._current_state_copy();

        # With probability epsilon choose a random action:
        if self._epsilon >= random.random():
            a = random.randint(0, len(self._actions) - 1)
        else:
            a = self._evaluator.best_action(s)
        self._actions_stats[a] += 1

        # make action and get the reward
        r = self._game.make_action(self._actions[a], self._skiprate + 1)
        r = np.float32(r)
        if self._shaping_on:
            sr = np.float32(doom_fixed_to_double(self._game.get_game_variable(GameVariable.USER1)))
            r += sr - self._last_shaping_reward
            self._last_shaping_reward = sr
        r = r * self._reward_scale
        if self._max_reward:
            r = min(r, self._max_reward)

        # update state s2 accordingly
        if self._game.is_episode_finished():
            # terminal state
            s2 = None
            self._transitions.add_transition(s, a, s2, r, terminal=True)
        else:
            self._update_state()
            # copy is not needed here cuase add transition copies it anyway
            s2 = self._current_state()
            self._transitions.add_transition(s, a, s2, r)

        # Perform q-learning once for a while
        if self._steps % self._update_pattern[0] == 0 and self._steps > self._batchsize:
            for i in range(self._update_pattern[1]):
                self.learn_batch()

    # Adds a transition to the bank.
    def add_transition(self, s,  a, s2, r, terminal):
        self._transitions.add_transition(s, a, s2, r, terminal)

    def learn_batch(self):
        self._evaluator.learn(self._transitions.get_sample())

    # Runs a single episode in current mode. It ignores the mode if learn==true/false
    def run_episode(self, learn=None):
        self.new_episode()
        if self.learning_mode and learn != False:
            self._update_state()
            while not self._game.is_episode_finished():
                self.make_learning_step()
        else:
            while not self._game.is_episode_finished():
                self.make_step()

        return np.float32(self._game.get_total_reward())

    # Utility stuff
    def get_actions_stats(self, clear=False, norm=True):
        stats = self._actions_stats.copy()
        if norm:
            stats = stats / np.float32(self._actions_stats.sum())
            stats[stats == 0.0] = -1
            stats = np.around(stats, 3)

        if clear:
            self._actions_stats.fill(0)
        return stats

    def get_steps(self):
        return self._steps

    def get_epsilon(self):
        return self._epsilon

    def get_network(self):
        return self._evaluator.get_network()

    def set_epsilon(self, eps):
        self._epsilon = eps

    def set_skiprate(self, skiprate):
        self._skiprate = skiprate

    def get_skiprate(self):
        return self._skiprate

    # Saves network weights to a file
    def save_params(self, filename, quiet=False):
        if not quiet:
            print "Saving network weights to " + filename + "..."
        self._prepare_for_save()
        params = get_all_param_values(self._evaluator.get_network())
        pickle.dump(params, open(filename, "wb"))
        if not quiet:
            print "Saving finished."

    # Loads network weights from the file
    def load_params(self, filename, quiet=False):
        if not quiet:
            print "Loading network weights from " + filename + "..."
        params = pickle.load(open(filename, "rb"))
        set_all_param_values(self._evaluator.get_network(), params)
        if not quiet:
            print "Loading finished."

            # Loads the whole engine with params from file

    @staticmethod
    def load(game, filename, quiet=False):
        if not quiet:
            print "Loading qengine from " + filename + "..."

        params = pickle.load(open(filename, "rb"))

        qengine_args = params[0]
        network_params = params[1]

        steps = qengine_args = qengine_args["steps"]
        epsilon = qengine_args = qengine_args["epsilon"]
        del(qengine_args["epsilon"])
        del(qengine_args["steps"])

        qengine_args["game"] = game
        qengine = QEngine(**qengine_args)
        set_all_param_values(qengine._evaluator.get_network(), network_params)
        if not quiet:
            print "Loading finished."
            qengine._steps = steps
            qengine._epsilon = epsilon
        return qengine

    # Saves the whole engine with params to a file
    def save(self, filename, quiet=False):
        if not quiet:
            print "Saving qengine to " + filename + "..."
        self._prepare_for_save()
        network_params = get_all_param_values(self._evaluator.get_network())
        params = [self._qengine_args, network_params]
        pickle.dump(params, open(filename, "wb"))
        if not quiet:
            print "Saving finished."
