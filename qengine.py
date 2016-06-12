import itertools as it
import pickle
import random
from math import log, floor
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


def initialize_doom(config_file, grayscale=True):
    doom = DoomGame()
    doom.load_config("common.cfg")
    doom.load_config(config_file)
    if grayscale:
        doom.set_screen_format(ScreenFormat.GRAY8)
    print "Initializing DOOM ..."
    doom.init()
    print "DOOM initialized."
    return doom


class QEngine:
    def __init__(self, **kwargs):
        self.setup = kwargs
        self._initialize(**kwargs)
        if "game" in kwargs:
            del kwargs["game"]

    def _prepare_for_save(self):
        self.setup["epsilon"] = self.epsilon
        self.setup["steps"] = self.steps
        self.setup["skiprate"] = self.skiprate

    # TODO why the fuck isn't it in init?
    def _initialize(self, game=None, network_args=None, actions=None, name=None, net_type="dqn",
                    reshaped_x=None,
                    reshaped_y=None,
                    skiprate=3,
                    history_length=4,
                    batchsize=64,
                    update_pattern=(1, 1),
                    replay_memory_size=10000,
                    backprop_start_step=10000,
                    start_epsilon=1.0,
                    end_epsilon=0.1,
                    epsilon_decay_start_step=50000,
                    epsilon_decay_steps=100000,
                    reward_scale=1.0,
                    melt_steps=10000,

                    shaping_on=False,
                    count_states=False,
                    use_game_variables=True,
                    remember_n_actions=4,
                    one_hot=False,

                    misc_scale=None,
                    results_file=None,
                    params_file=None,
                    config_file=None

                    ):

        if game is not None:
            self.game = game
            self.config_file = None
        elif config_file is not None:
            self.config_file = config_file
            self.game = initialize_doom(self.config_file)
        else:
            raise Exception("No game, no config file. Dunno how to initialize doom.")
        if network_args is None:
            network_args = dict()
        if count_states is not None:
            self._count_states = bool(count_states)

        self.name = name
        self.reward_scale = reward_scale

        self.batchsize = batchsize
        self.history_length = max(history_length, 1)
        self.update_pattern = update_pattern
        self.epsilon = max(min(start_epsilon, 1.0), 0.0)
        self.end_epsilon = min(max(end_epsilon, 0.0), self.epsilon)
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_stride = (self.epsilon - end_epsilon) / epsilon_decay_steps
        self.epsilon_decay_start = epsilon_decay_start_step
        self.skiprate = max(skiprate, 0)
        self.shaping_on = shaping_on
        self.steps = 0
        self.melt_steps = melt_steps
        self.backprop_start_step = max(backprop_start_step, batchsize)
        self.one_hot = one_hot

        if results_file:
            self.results_file = results_file
        else:
            self.results_file = "results/" + name + ".res"
        if params_file:
            self.params_file = params_file
        else:
            self.params_file = "params/" + name + ".res"

        if self.game.get_available_game_variables_size() > 0 and use_game_variables:
            self.use_game_variables = True
        else:
            self.use_game_variables = False

        if self.shaping_on:
            self.last_shaping_reward = 0

        self.learning_mode = True

        if actions is None:
            self.actions = generate_default_actions(self.game)
        else:
            self.actions = actions

        self.actions_num = len(self.actions)

        self.actions_stats = np.zeros([self.actions_num], np.int)

        # changes img_shape according to the history size
        self._channels = self.game.get_screen_channels()
        if self.history_length > 1:
            self._channels *= self.history_length

        if reshaped_x is None:
            x = self.game.get_screen_width()
            y = self.game.get_screen_height()
            scale_x = scale_y = 1.0
        else:
            x = reshaped_x
            scale_x = float(x) / self.game.get_screen_width()

            if reshaped_y is None:
                y = int(self.game.get_screen_height() * scale_x)
                scale_y = scale_x
            else:
                y = reshaped_y
                scale_y = float(y) / self.game.get_screen_height()

        img_shape = [self._channels, y, x]

        # TODO check if it is slow (it seems that no)
        if scale_x == 1 and scale_y == 1:
            def convert(img):
                img = img.astype(np.float32) / 255.0
                return img
        else:
            def convert(img):
                img = img.astype(np.float32) / 255.0
                new_image = np.ndarray([img.shape[0], y, x], dtype=img.dtype)
                for i in xrange(img.shape[0]):
                    # new_image[i] = skimage.transform.resize(img[i], (y,x), preserve_range=True)
                    new_image[i] = cv2.resize(img[i], (x, y), interpolation=cv2.INTER_AREA)
                return new_image
        self.convert_image_fun = convert

        if self.use_game_variables:
            single_state_misc_len = self.game.get_available_game_variables_size() + int(self._count_states)
        else:
            single_state_misc_len = int(self._count_states)
        self.single_state_misc_len = single_state_misc_len

        self.remember_n_actions = remember_n_actions
        if remember_n_actions > 0:
            self.remember_n_actions = remember_n_actions
            if self.one_hot:
                self._action_len = int(2 ** floor(log(len(self.actions), 2)))
            else:
                self._action_len = len(self.actions[0])
            self.last_action = np.zeros([self._action_len], dtype=np.float32)
            self._last_n_actions = np.zeros([remember_n_actions * self._action_len], dtype=np.float32)
            self._total_misc_len = single_state_misc_len * self.history_length + len(self._last_n_actions)

        else:
            self._total_misc_len = single_state_misc_len * self.history_length

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
        self._transitions = ReplayMemory(state_format, replay_memory_size, batchsize)

        network_args["state_format"] = state_format
        network_args["actions_number"] = len(self.actions)

        if net_type in ("dqn", None, ""):
            self._evaluator = DQN(**network_args)
        elif net_type in ["duelling", "dueling"]:
            self._evaluator = DuelingDQN(**network_args)
        else:
            print "Unsupported evaluator type."
            exit(1)
            # TODO throw. . .?

        self._current_image_state = np.zeros(img_shape, dtype=np.float32)

    def _update_state(self):
        raw_state = self.game.get_state()
        img = self.convert_image_fun(raw_state.image_buffer)
        state_misc = None

        if self.single_state_misc_len > 0:
            state_misc = self._state_misc_buffer

            if self.use_game_variables:
                game_variables = raw_state.game_variables.astype(np.float32)
                state_misc[0:len(game_variables)] = game_variables

            if self._count_states:
                state_misc[-1] = raw_state.number

            if self._misc_scale is not None:
                state_misc = state_misc * self._misc_scale

        if self.history_length > 1:
            pure_channels = self._channels / self.history_length
            self._current_image_state[0:-pure_channels] = self._current_image_state[pure_channels:]
            self._current_image_state[-pure_channels:] = img

            if self.single_state_misc_len > 0:
                misc_len = len(state_misc)
                hist = self.history_length
                self._current_misc_state[0:(hist - 1) * misc_len] = self._current_misc_state[misc_len:hist * misc_len]

                self._current_misc_state[(hist - 1) * misc_len:hist * misc_len] = state_misc

        else:
            self._current_image_state[:] = img
            if self.single_state_misc_len > 0:
                self._current_misc_state[0:len(state_misc)] = state_misc

        if self.remember_n_actions:
            self._last_n_actions[:-self._action_len] = self._last_n_actions[self._action_len:]

            self._last_n_actions[-self._action_len:] = self.last_action
            self._current_misc_state[-len(self._last_n_actions):] = self._last_n_actions

    def new_episode(self, update_state=False):
        self.game.new_episode()
        self.reset_state()
        self.last_shaping_reward = 0
        if update_state:
            self._update_state()

    def set_last_action(self, index):
        if self.one_hot:
            self.last_action.fill(0)
            self.last_action[index] = 1
        else:
            self.last_action[:] = self.actions[index]

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
            if self.remember_n_actions > 0:
                self.set_last_action(0)
                self._last_n_actions.fill(0)

    def make_step(self):
        self._update_state()
        # TODO Check if not making the copy still works
        a = self._evaluator.estimate_best_action(self._current_state_copy())
        self.actions_stats[a] += 1
        self.game.make_action(self.actions[a], self.skiprate + 1)
        if self.remember_n_actions:
            self.set_last_action(a)

    def make_sleep_step(self, sleep_time=1 / 35.0):
        self._update_state()
        a = self._evaluator.estimate_best_action(self._current_state_copy())
        self.actions_stats[a] += 1

        self.game.set_action(self.actions[a])
        if self.remember_n_actions:
            self.set_last_action(a)
        for i in xrange(self.skiprate):
            self.game.advance_action(1, False, True)
            sleep(sleep_time)
        self.game.advance_action()
        sleep(sleep_time)

    # Performs a learning step according to epsilon-greedy policy.
    # The step spans self._skiprate +1 actions.
    def make_learning_step(self):
        self.steps += 1
        # epsilon decay
        if self.steps > self.epsilon_decay_start and self.epsilon > self.end_epsilon:
            self.epsilon = max(self.epsilon - self.epsilon_decay_stride, 0)

            # Copy because state will be changed in a second
        s = self._current_state_copy();

        # With probability epsilon choose a random action:
        if self.epsilon >= random.random():
            a = random.randint(0, len(self.actions) - 1)
        else:
            a = self._evaluator.estimate_best_action(s)
        self.actions_stats[a] += 1

        # make action and get the reward
        if self.remember_n_actions:
            self.set_last_action(a)

        r = self.game.make_action(self.actions[a], self.skiprate + 1)
        r = np.float32(r)
        if self.shaping_on:
            sr = np.float32(doom_fixed_to_double(self.game.get_game_variable(GameVariable.USER1)))
            r += sr - self.last_shaping_reward
            self.last_shaping_reward = sr

        if self.reward_scale is not None:
            r *= self.reward_scale

        # update state s2 accordingly
        if self.game.is_episode_finished():
            # terminal state
            s2 = None
            self._transitions.add_transition(s, a, s2, r, terminal=True)
        else:
            self._update_state()
            s2 = self._current_state()
            self._transitions.add_transition(s, a, s2, r, terminal=False)

        # Perform q-learning once for a while
        if self._transitions.size >= self.backprop_start_step and self.steps % self.update_pattern[0] == 0:
            for a in xrange(self.update_pattern[1]):
                self._evaluator.learn(self._transitions.get_sample())

        # Melt the network sometimes
        if self.steps % self.melt_steps == 0:
            self._evaluator.melt()

    # Adds a transition to the bank.
    def add_transition(self, s, a, s2, r, terminal):
        self._transitions.add_transition(s, a, s2, r, terminal)

    # Runs a single episode in current mode. It ignores the mode if learn==true/false
    def run_episode(self, sleep_time=0):
        self.new_episode()
        if sleep_time == 0:
            while not self.game.is_episode_finished():
                self.make_step()
        else:
            while not self.game.is_episode_finished():
                self.make_sleep_step(sleep_time)

        return np.float32(self.game.get_total_reward())

    # Utility stuff
    def get_actions_stats(self, clear=False, norm=True):
        stats = self.actions_stats.copy()
        if norm:
            stats = stats / np.float32(self.actions_stats.sum())
            stats[stats == 0.0] = -1
            stats = np.around(stats, 3)

        if clear:
            self.actions_stats.fill(0)
        return stats

    def get_steps(self):
        return self.steps

    def get_epsilon(self):
        return self.epsilon

    def get_network(self):
        return self._evaluator.network

    def set_epsilon(self, eps):
        self.epsilon = eps

    def set_skiprate(self, skiprate):
        self.skiprate = max(skiprate, 0)

    def get_skiprate(self):
        return self.skiprate

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
        set_all_param_values(self._evaluator.frozen_network, params)

        if not quiet:
            print "Loading finished."

            # Loads the whole engine with params from file

    def get_network_architecture(self):
        return get_all_param_values(self.get_network())

    def print_setup(self):
        print "\nNetwork architecture:"
        for p in self.get_network_architecture():
            print p.shape
        print "\n*** Engine setup ***"
        for k in self.setup.keys():
            if k == "network_args":
                print"network_args:"
                net_args = self.setup[k]
                for k2 in net_args.keys():
                    print "\t", k2, ":", net_args[k2]
            else:
                print k, ":", self.setup[k]

    @staticmethod
    def load(filename, game=None, config_file=None, quiet=False):
        if not quiet:
            print "Loading qengine from " + filename + "..."

        params = pickle.load(open(filename, "rb"))

        qengine_args = params[0]
        network_weights = params[1]

        steps = qengine_args["steps"]
        epsilon = qengine_args["epsilon"]
        del (qengine_args["epsilon"])
        del (qengine_args["steps"])
        if game is None:
            if config_file is not None:
                game = initialize_doom(config_file)
                qengine_args["config_file"] = config_file
            elif qengine_args["config_file"] is not None:
                game = initialize_doom(qengine_args["config_file"])
            else:
                raise Exception("No game, no config file. Dunno how to initialize doom.")
        else:
            qengine_args["config_file"] = None

        qengine_args["game"] = game
        qengine = QEngine(**qengine_args)
        set_all_param_values(qengine._evaluator.network, network_weights)
        set_all_param_values(qengine._evaluator.frozen_network, network_weights)

        if not quiet:
            print "Loading finished."
            qengine.steps = steps
            qengine.epsilon = epsilon
        return qengine

    # Saves the whole engine with params to a file
    def save(self, filename=None, quiet=False):
        if filename is None:
            filename = self.params_file
        if not quiet:
            print "Saving qengine to " + filename + "..."
        self._prepare_for_save()
        network_params = get_all_param_values(self._evaluator.network)
        params = [self.setup, network_params]
        pickle.dump(params, open(filename, "wb"))
        if not quiet:
            print "Saving finished."
