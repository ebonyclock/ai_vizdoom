from collections import OrderedDict

import lasagne
import lasagne.layers as ls
import numpy as np
import theano
import theano.tensor as tensor
import theano.tensor as T
from lasagne.nonlinearities import rectify
from lasagne.updates import get_or_compute_grads


def deepmind_rmsprop(loss_or_grads, params, learning_rate=0.00025,
                     rho=0.95, epsilon=0.01):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)

        acc_grad = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        acc_grad_new = rho * acc_grad + (1 - rho) * grad

        acc_rms = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                broadcastable=param.broadcastable)
        acc_rms_new = rho * acc_rms + (1 - rho) * grad ** 2

        updates[acc_grad] = acc_grad_new
        updates[acc_rms] = acc_rms_new

        updates[param] = (param - learning_rate *
                          (grad /
                           T.sqrt(acc_rms_new - acc_grad_new ** 2 + epsilon)))

    return updates


class DQN:
    def __init__(self, state_format, actions_number, architecture=None, gamma=0.99, learning_rate=0.00025):
        self._inputs = dict()
        if architecture is None:
            architecture = dict()

        self._loss_history = []
        self._misc_state_included = (state_format["s_misc"] > 0)
        self._gamma = np.float64(gamma)

        self._inputs["S0"] = tensor.tensor4("S0")
        self._inputs["S1"] = tensor.tensor4("S1")
        self._inputs["A"] = tensor.vector("Action", dtype="int32")
        self._inputs["R"] = tensor.vector("Reward")
        self._inputs["Nonterminal"] = tensor.vector("Nonterminal", dtype="int8")
        if self._misc_state_included:
            self._inputs["S0_misc"] = tensor.matrix("S0_misc")
            self._inputs["S1_misc"] = tensor.matrix("S1_misc")
            self._misc_len = state_format["s_misc"]
        else:
            self._misc_len = None

        # save it for the evaluation reshape
        # TODO get rid of this?
        self._single_image_input_shape = (1,) + tuple(state_format["s_img"])

        architecture["img_input_shape"] = (None,) + tuple(state_format["s_img"])
        architecture["misc_len"] = self._misc_len
        architecture["output_size"] = actions_number

        if self._misc_state_included:
            self.network = self._initialize_network(img_input=self._inputs["S0"], misc_input=self._inputs["S0_misc"],
                                                    **architecture)
            self.frozen_network = self._initialize_network(img_input=self._inputs["S1"],
                                                           misc_input=self._inputs["S1_misc"], **architecture)
        else:

            self.network = self._initialize_network(img_input=self._inputs["S0"], **architecture)
            self.frozen_network = self._initialize_network(img_input=self._inputs["S1"], **architecture)

        # print "Network initialized."
        self._learning_rate = learning_rate
        self._compile()

    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):

        # weights_init = lasagne.init.GlorotUniform("relu")
        weights_init = lasagne.init.HeNormal("relu")

        network = ls.InputLayer(shape=img_input_shape, input_var=img_input)

        network = ls.Conv2DLayer(network, num_filters=32, filter_size=8, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(.1), stride=4)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=4, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(.1), stride=2)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=3, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(.1), stride=1)

        if self._misc_state_included:
            network = ls.FlattenLayer(network)
            misc_input_layer = ls.InputLayer(shape=(None, misc_len), input_var=misc_input)
            network = ls.ConcatLayer([network, misc_input_layer])

        network = ls.DenseLayer(network, 512, nonlinearity=rectify,
                                W=weights_init, b=lasagne.init.Constant(.1))

        network = ls.DenseLayer(network, output_size, nonlinearity=None, b=lasagne.init.Constant(.1))
        return network

    def _compile(self):

        a = self._inputs["A"]
        r = self._inputs["R"]

        q = ls.get_output(self.network, deterministic=False)
        deterministic_q = ls.get_output(self.network, deterministic=True)

        q2 = tensor.max(ls.get_output(self.frozen_network, deterministic=True), axis=1, keepdims=True)

        nonterminal = self._inputs["Nonterminal"]
        target_q = r + self._gamma * nonterminal * q2

        # Loss
        abs_err = abs(q[tensor.arange(q.shape[0]), a] - target_q)
        quadratic_part = tensor.minimum(abs_err, 1)
        linear_part = abs_err - quadratic_part
        loss = (0.5 * quadratic_part ** 2 + linear_part).mean()

        params = ls.get_all_params(self.network, trainable=True)

        # updates = lasagne.updates.rmsprop(loss, params, self._learning_rate, rho=0.95)
        updates = deepmind_rmsprop(loss, params, self._learning_rate)

        # TODO find out why this mode causes problems with misc vector
        # mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        mode = None

        s0_img = self._inputs["S0"]
        s1_img = self._inputs["S1"]
        if self._misc_state_included:
            s0_misc = self._inputs["S0_misc"]
            s1_misc = self._inputs["S1_misc"]
            self._learn = theano.function([s0_img, s0_misc, s1_img, s1_misc, a, r, nonterminal], loss,
                                          updates=updates, mode=mode, name="learn_fn")
            self._evaluate = theano.function([s0_img, s0_misc], deterministic_q, mode=mode,
                                             name="eval_fn")
        else:
            self._learn = theano.function([s0_img, s1_img, a, r, nonterminal], loss, updates=updates, mode=mode,
                                          name="learn_fn")
            self._evaluate = theano.function([s0_img], deterministic_q, mode=mode, name="eval_fn")

    def learn(self, transitions):
        # Learning approximation: Q(s1,t+1) = r + nonterminal *Q(s2,t)

        X = transitions["s1_img"]
        X2 = transitions["s2_img"]

        if self._misc_state_included:
            X_misc = transitions["s1_misc"]
            X2_misc = transitions["s2_misc"]
            loss = self._learn(X, X_misc, X2, X2_misc, transitions["a"], transitions["r"], transitions["nonterminal"])
        else:
            loss = self._learn(X, X2, transitions["a"], transitions["r"], transitions["nonterminal"])

        self._loss_history.append(loss)

    def estimate_best_action(self, state):
        if self._misc_state_included:
            qvals = self._evaluate(state[0].reshape(self._single_image_input_shape),
                                   state[1].reshape(1, self._misc_len))
            a = np.argmax(qvals)
        else:
            qvals = self._evaluate(state[0].reshape(self._single_image_input_shape))
            a = np.argmax(qvals)
        return a

    def get_mean_loss(self, clear=True):
        m = np.mean(self._loss_history)
        if clear:
            self._loss_history = []
        return m

    def get_network(self):
        return self.network

    def melt(self):
        ls.set_all_param_values(self.frozen_network, ls.get_all_param_values(self.network))

