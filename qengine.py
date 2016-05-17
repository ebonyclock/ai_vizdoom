import itertools as it
import pickle
import random
from time import sleep
from vizdoom import *

import cv2
from lasagne.layers import get_all_param_values
from lasagne.layers import set_all_param_values

from evaluators import *
from replay_memory import ReplayMemory


def generate_default_actions(the_game):
    n = the_game.get_available_buttons_size()
    actions = []
    for perm in it.product([0, 1], repeat=n):
        actions.append(list(perm))
    return actions


class QEngine:
    def __init__(self, **kwargs):
        self.setup = kwargs
        self._initialize(**kwargs)
        kwargs["game"] = None

    def _prepare_for_save(self):
        self.setup["epsilon"] = self._epsilon
        self.setup["steps"] = self._steps
        self.setup["skiprate"] = self._skiprate

    def _initialize(self, game, network_args, history_length=1, batchsize=64,
                    update_pattern=(4, 4),
                    bank_capacity=10000, start_epsilon=1.0, end_epsilon=0.1, epsilon_decay_start_step=100000,
                    epsilon_decay_steps=100000,
                    reward_scale=1.0, misc_scale=None, max_reward=None, reshaped_x=120, skiprate=0,
                    shaping_on=False, count_states=False, actions=None, name=None, type="cnn", frozen_steps=1000,
                    freeze=False, remember_n_actions=0):

        if count_states is not None:
            self._count_states = bool(count_states)

        if max_reward is not None:
            self._max_reward = abs(np.float32(max_reward))
        else:
            self._max_reward = None

        self.name = name
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
        self._frozen_steps = frozen_steps
        self._freeze = freeze

        if self._shaping_on:
            self._last_shaping_reward = 0

        self.learning_mode = True

        if actions is None:
            self._actions = generate_default_actions(game)
        else:
            self._actions = actions

        self._actions_num = len(self._actions)
        self._actions_stats = np.zeros([self._actions_num], np.int)

        # changes img_shape according to the history size
        self._channels = game.get_screen_channels()
        if self._history_length > 1:
            self._channels *= self._history_length

        self._scale = float(reshaped_x) / game.get_screen_width()
        y = int(game.get_screen_height() * self._scale)
        x = reshaped_x

        img_shape = [self._channels, y, x]

        if self._scale == 1:

            def convert(img):
                img = img.astype(np.float32) / 255.0
                return img
        else:
            def convert(img):
                img = img.astype(np.float32) / 255.0
                new_image = np.ndarray([img.shape[0], y, x], dtype=np.float32)
                for i in range(img.shape[0]):
                    new_image[i] = cv2.resize(img[i], (x, y))
                return new_image

        self._convert_image = convert

        single_state_misc_len = game.get_available_game_variables_size() + self._count_states
        self._single_state_misc_len = single_state_misc_len
        self._remember_n_actions = remember_n_actions
        if remember_n_actions > 0:
            self._remember_n_actions = remember_n_actions
            self._action_len = len(self._actions[0])
            self._last_n_actions = np.zeros([remember_n_actions * self._action_len], dtype=np.float32)
            self._total_misc_len = single_state_misc_len * self._history_length + len(self._last_n_actions)
            self._last_action_index = 0
        else:
            self._total_misc_len = single_state_misc_len * self._history_length

        if self._total_misc_len > 0:
            self._misc_state_included = True
            self._current_misc_state = np.zeros(self._total_misc_len, dtype=np.float32)
            if single_state_misc_len > 0:
                self._state_misc_buffer = np.zeros(single_state_misc_len, dtype=np.float32)
                if misc_scale is not None:
                    self._misc_scale = np.array(misc_scale, dtype=np.float32)
                else:
                    self._misc_scale = None
        else:
            self._misc_state_included = False

        state_format = dict()
        state_format["s_img"] = img_shape
        state_format["s_misc"] = self._total_misc_len
        self._transitions = ReplayMemory(state_format, bank_capacity, batchsize)

        network_args["state_format"] = state_format
        network_args["actions_number"] = len(self._actions)
        network_args["freeze"] = freeze

        if type in ("cnn", None, ""):
            self._evaluator = CNNEvaluator(**network_args)
        elif type == "cnn_mem":
            network_args["architecture"]["memory"] = self._history_length
            self._evaluator = CNNEvaluator_mem(**network_args)
        elif type == "mlp":
            self._evaluator = MLPEvaluator(**network_args)
        else:
            print "Unsupported evaluator type specified"

        self._current_image_state = np.zeros(img_shape, dtype=np.float32)

    def _update_state(self):
        raw_state = self._game.get_state()
        img = self._convert_image(raw_state.image_buffer)
        state_misc = None

        if self._single_state_misc_len > 0:
            state_misc = self._state_misc_buffer

            if self._count_states:
                state_misc[0:-1] = np.float32(raw_state.game_variables)
                state_misc[-1] = raw_state.number
            else:
                state_misc[:] = np.float32(raw_state.game_variables)

            if self._misc_scale is not None:
                state_misc = state_misc * self._misc_scale

        if self._history_length > 1:
            pure_channels = self._channels / self._history_length
            self._current_image_state[0:-pure_channels] = self._current_image_state[pure_channels:]
            self._current_image_state[-pure_channels:] = img

            if self._single_state_misc_len > 0:
                self._current_misc_state = np.roll(self._current_misc_state, -len(state_misc))
                a = len(self._current_misc_state)
                self._current_misc_state[a - len(state_misc):a] = state_misc

        else:
            self._current_image_state[:] = img
            if self._single_state_misc_len > 0:
                self._current_misc_state[0:len(state_misc)] = state_misc

        if self._remember_n_actions:
            np.roll(self._last_n_actions, -self._action_len)
            self._last_n_actions[-self._action_len:] = self._actions[self._last_action_index]
            self._current_misc_state[-len(self._last_n_actions):] = self._last_n_actions

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
            if self._remember_n_actions > 0:
                self._last_n_actions.fill(0)

    def make_step(self):
        self._update_state()
        # TODO Check if not making the copy still works
        a = self._evaluator.best_action(self._current_state_copy())
        self._actions_stats[a] += 1
        self._game.make_action(self._actions[a], self._skiprate + 1)
        self._last_action_index = a

    def make_sleep_step(self, sleep_time=1/35.0):
        self._update_state()
        a = self._evaluator.best_action(self._current_state_copy())
        self._actions_stats[a] += 1

        self._game.set_action(self._actions[a])
        self._last_action_index = a
        for i in range(self._skiprate):
            self._game.advance_action(1, False, True)
            sleep(sleep_time)
        self._game.advance_action()
        sleep(sleep_time)

    # Performs a learning step according to epsilon-greedy policy.
    # The step spans self._skiprate +1 actions.
    def make_learning_step(self):
        self._steps += 1
        # epsilon decay
        if self._steps > self._epsilon_decay_start and self._epsilon > self._end_epsilon:
            self._epsilon = max(self._epsilon - self._epsilon_decay_stride, 0)

            # Copy because state will be changed in a second
        s = self._current_state_copy();

        # With probability epsilon choose a random action:
        if self._epsilon >= random.random():
            a = random.randint(0, len(self._actions) - 1)
        else:
            a = self._evaluator.best_action(s)
        self._actions_stats[a] += 1

        # make action and get the reward
        self._last_action_index = a
        r = self._game.make_action(self._actions[a], self._skiprate + 1)
        r = np.float32(r)
        if self._shaping_on:
            sr = np.float32(doom_fixed_to_double(self._game.get_game_variable(GameVariable.USER1)))
            r += sr - self._last_shaping_reward
            self._last_shaping_reward = sr

        if self._max_reward:
            r = np.clip(r, -self._max_reward, self._max_reward)

        r *= self._reward_scale

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
        if self._transitions.get_size() > self._batchsize and self._steps % self._update_pattern[0] == 0:
            for i in range(self._update_pattern[1]):
                self.learn_batch()

        # Melt the network sometimes
        if self._freeze:
            if (self._steps + 1) % self._frozen_steps:
                self._evaluator.melt()

    # Adds a transition to the bank.
    def add_transition(self, s, a, s2, r, terminal):
        self._transitions.add_transition(s, a, s2, r, terminal)

    def learn_batch(self):
        self._evaluator.learn(self._transitions.get_sample())

    # Runs a single episode in current mode. It ignores the mode if learn==true/false
    def run_episode(self, sleep_time=0):
        self.new_episode()
        if sleep_time==0:
            while not self._game.is_episode_finished():
                self.make_step()
        else:
            while not self._game.is_episode_finished():
                self.make_sleep_step(sleep_time)

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
        return self._evaluator.network

    def set_epsilon(self, eps):
        self._epsilon = eps

    def set_skiprate(self, skiprate):
        self._skiprate = max(skiprate, 0)

    def get_skiprate(self):
        return self._skiprate

    # Saves network weights to a file
    def save_params(self, filename, quiet=False):
        if not quiet:
            print "Saving network weights to " + filename + "..."
        self._prepare_for_save()
        params = get_all_param_values(self._evaluator.network)
        pickle.dump(params, open(filename, "wb"))
        if not quiet:
            print "Saving finished."

    # Loads network weights from the file
    def load_params(self, filename, quiet=False):
        if not quiet:
            print "Loading network weights from " + filename + "..."
        params = pickle.load(open(filename, "rb"))
        set_all_param_values(self._evaluator.network, params)
        if self._freeze:
            set_all_param_values(self._evaluator.frozen_network, params)
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

        steps = qengine_args["steps"]
        epsilon = qengine_args["epsilon"]
        del (qengine_args["epsilon"])
        del (qengine_args["steps"])

        qengine_args["game"] = game

        qengine = QEngine(**qengine_args)
        set_all_param_values(qengine._evaluator.network, network_params)
        if qengine._freeze:
            set_all_param_values(qengine._evaluator.frozen_network, network_params)

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
        network_params = get_all_param_values(self._evaluator.network)
        params = [self.setup, network_params]
        pickle.dump(params, open(filename, "wb"))
        if not quiet:
            print "Saving finished."
