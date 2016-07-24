from collections import OrderedDict
import lasagne
import lasagne.layers as ls
import numpy as np
import theano
import theano.tensor as tensor
import theano.tensor as T
from lasagne.nonlinearities import rectify, sigmoid
from lasagne.updates import get_or_compute_grads


class DuellingMergeLayer(ls.MergeLayer):
    def __init__(self, incomings, **kwargs):
        ls.MergeLayer.__init__(self, incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        m = tensor.mean(inputs[0], axis=1, keepdims=True)
        sv = tensor.addbroadcast(inputs[1], 1)
        return inputs[0] + sv - m


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
    def __init__(self, state_format, actions_number, gamma=0.99, learning_rate=0.00025, ddqn=False, **kwargs):
        self.inputs = dict()
        self.learning_rate = learning_rate
        architecture = kwargs

        self.loss_history = []
        self.misc_state_included = (state_format["s_misc"] > 0)
        self.gamma = np.float64(gamma)

        self.inputs["S0"] = tensor.tensor4("S0")
        self.inputs["S1"] = tensor.tensor4("S1")
        self.inputs["A"] = tensor.ivector("Action")
        self.inputs["R"] = tensor.vector("Reward")
        self.inputs["Nonterminal"] = tensor.bvector("Nonterminal")
        if self.misc_state_included:
            self.inputs["S0_misc"] = tensor.matrix("S0_misc")
            self.inputs["S1_misc"] = tensor.matrix("S1_misc")
            self.misc_len = state_format["s_misc"]
        else:
            self.misc_len = None

        # save it for the evaluation reshape
        # TODO get rid of this?
        self.single_image_input_shape = (1,) + tuple(state_format["s_img"])

        architecture["img_input_shape"] = (None,) + tuple(state_format["s_img"])
        architecture["misc_len"] = self.misc_len
        architecture["output_size"] = actions_number

        if self.misc_state_included:
            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"],
                                                                     misc_input=self.inputs["S0_misc"],
                                                                     **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                misc_input=self.inputs["S1_misc"],
                                                                                **architecture)
        else:

            self.network, input_layers, _ = self._initialize_network(img_input=self.inputs["S0"], **architecture)
            self.frozen_network, _, alternate_inputs = self._initialize_network(img_input=self.inputs["S1"],
                                                                                **architecture)

        self.alternate_input_mappings = {}
        for layer, input in zip(input_layers, alternate_inputs):
            self.alternate_input_mappings[layer] = input

        # print "Network initialized."
        self._compile(ddqn)

    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):

        input_layers = []
        inputs = [img_input]
        # weights_init = lasagne.init.GlorotUniform("relu")
        weights_init = lasagne.init.HeNormal("relu")

        network = ls.InputLayer(shape=img_input_shape, input_var=img_input)
        input_layers.append(network)
        network = ls.Conv2DLayer(network, num_filters=32, filter_size=8, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=4)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=4, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=2)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=3, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=1)

        if self.misc_state_included:
            inputs.append(misc_input)
            network = ls.FlattenLayer(network)
            misc_input_layer = ls.InputLayer(shape=(None, misc_len), input_var=misc_input)
            input_layers.append(misc_input_layer)
            if "additional_misc_layer" in kwargs:
                misc_input_layer = ls.DenseLayer(misc_input_layer, int(kwargs["additional_misc_layer"]),
                                                 nonlinearity=rectify,
                                                 W=weights_init, b=lasagne.init.Constant(0.1))

            network = ls.ConcatLayer([network, misc_input_layer])

        network = ls.DenseLayer(network, 512, nonlinearity=rectify,
                                W=weights_init, b=lasagne.init.Constant(0.1))

        network = ls.DenseLayer(network, output_size, nonlinearity=None, b=lasagne.init.Constant(.1))
        return network, input_layers, inputs

    @staticmethod
    def build_loss_expression(predicted, target):

        abs_err = abs(predicted - target)
        quadratic_part = tensor.minimum(abs_err, 1)
        linear_part = abs_err - quadratic_part
        loss = (0.5 * quadratic_part ** 2 + linear_part)
        return loss

    def _compile(self, ddqn):

        a = self.inputs["A"]
        r = self.inputs["R"]
        nonterminal = self.inputs["Nonterminal"]

        q = ls.get_output(self.network, deterministic=True)

        if ddqn:
            q2 = ls.get_output(self.network, deterministic=True, inputs=self.alternate_input_mappings)
            q2_action_ref = tensor.argmax(q2, axis=1)

            q2_frozen = ls.get_output(self.frozen_network, deterministic=True)
            q2_max = q2_frozen[tensor.arange(q2_action_ref.shape[0]), q2_action_ref]
        else:
            q2_max = tensor.max(ls.get_output(self.frozen_network, deterministic=True), axis=1)

        target_q = r + self.gamma * nonterminal * q2_max
        predicted_q = q[tensor.arange(q.shape[0]), a]

        loss = self.build_loss_expression(predicted_q, target_q).sum()
        params = ls.get_all_params(self.network, trainable=True)

        # updates = lasagne.updates.rmsprop(loss, params, self._learning_rate, rho=0.95)
        updates = deepmind_rmsprop(loss, params, self.learning_rate)

        # TODO does FAST_RUN speed anything up?
        mode = None  # "FAST_RUN"

        s0_img = self.inputs["S0"]
        s1_img = self.inputs["S1"]

        if self.misc_state_included:
            s0_misc = self.inputs["S0_misc"]
            s1_misc = self.inputs["S1_misc"]
            print "Compiling the training function..."
            self._learn = theano.function([s0_img, s0_misc, s1_img, s1_misc, a, r, nonterminal], loss,
                                          updates=updates, mode=mode, name="learn_fn")
            print "Compiling the evaluation function..."
            self._evaluate = theano.function([s0_img, s0_misc], q, mode=mode,
                                             name="eval_fn")
        else:
            print "Compiling the training function..."
            self._learn = theano.function([s0_img, s1_img, a, r, nonterminal], loss, updates=updates, mode=mode,
                                          name="learn_fn")
            print "Compiling the evaluation function..."
            self._evaluate = theano.function([s0_img], q, mode=mode, name="eval_fn")
        print "Network compiled."

    def learn(self, transitions):
        t = transitions
        if self.misc_state_included:
            loss = self._learn(t["s1_img"], t["s1_misc"], t["s2_img"], t["s2_misc"], t["a"], t["r"], t["nonterminal"])
        else:
            loss = self._learn(t["s1_img"], t["s2_img"], t["a"], t["r"], t["nonterminal"])
        self.loss_history.append(loss)

    def estimate_best_action(self, state):
        if self.misc_state_included:
            qvals = self._evaluate(state[0].reshape(self.single_image_input_shape),
                                   state[1].reshape(1, self.misc_len))
            a = np.argmax(qvals)
        else:
            qvals = self._evaluate(state[0].reshape(self.single_image_input_shape))
            a = np.argmax(qvals)
        return a

    def get_mean_loss(self, clear=True):
        m = np.mean(self.loss_history)
        if clear:
            self.loss_history = []
        return m

    def get_network(self):
        return self.network

    def melt(self):
        ls.set_all_param_values(self.frozen_network, ls.get_all_param_values(self.network))


class DuelingDQN(DQN):
    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]
        # weights_init = lasagne.init.GlorotUniform("relu")
        weights_init = lasagne.init.HeNormal("relu")

        network = ls.InputLayer(shape=img_input_shape, input_var=img_input)
        input_layers.append(network)
        network = ls.Conv2DLayer(network, num_filters=32, filter_size=8, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(.1), stride=4)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=4, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(.1), stride=2)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=3, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(.1), stride=1)

        if self.misc_state_included:
            inputs.append(misc_input)
            network = ls.FlattenLayer(network)
            misc_input_layer = ls.InputLayer(shape=(None, misc_len), input_var=misc_input)
            input_layers.append(misc_input_layer)
            if "additional_misc_layer" in kwargs:
                misc_input_layer = ls.DenseLayer(misc_input_layer, int(kwargs["additional_misc_layer"]),
                                                 nonlinearity=rectify,
                                                 W=weights_init, b=lasagne.init.Constant(0.1))
            network = ls.ConcatLayer([network, misc_input_layer])

        # Duelling here

        advanteges_branch = ls.DenseLayer(network, 256, nonlinearity=rectify,
                                          W=weights_init, b=lasagne.init.Constant(.1))
        advanteges_branch = ls.DenseLayer(advanteges_branch, output_size, nonlinearity=None,
                                          b=lasagne.init.Constant(.1))

        state_value_branch = ls.DenseLayer(network, 256, nonlinearity=rectify,
                                           W=weights_init, b=lasagne.init.Constant(.1))
        state_value_branch = ls.DenseLayer(state_value_branch, 1, nonlinearity=None,
                                           b=lasagne.init.Constant(.1))

        network = DuellingMergeLayer([advanteges_branch, state_value_branch])
        return network, input_layers, inputs


class DQNHealthFC(DQN):
    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]
        # weights_init = lasagne.init.GlorotUniform("relu")
        weights_init = lasagne.init.HeNormal("relu")

        network = ls.InputLayer(shape=img_input_shape, input_var=img_input)
        input_layers.append(network)
        network = ls.Conv2DLayer(network, num_filters=32, filter_size=8, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=4)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=4, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=2)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=3, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=1)
        network = ls.FlattenLayer(network)

        if self.misc_state_included:
            health_inputs = 4
            units_per_health_input = 100
            layers_for_merge = []
            for i in range(health_inputs):
                health_input_layer = ls.InputLayer(shape=(None, 1), input_var=misc_input[:, i:i + 1])

                health_layer = ls.DenseLayer(health_input_layer, units_per_health_input, nonlinearity=rectify,
                                             W=weights_init, b=lasagne.init.Constant(0.1))
                health_layer = ls.DenseLayer(health_layer, units_per_health_input, nonlinearity=rectify,
                                             W=weights_init, b=lasagne.init.Constant(0.1))

                inputs.append(misc_input[:, i:i + 1])
                input_layers.append(health_input_layer)
                layers_for_merge.append(health_layer)

            misc_input_layer = ls.InputLayer(shape=(None, misc_len - health_inputs),
                                             input_var=misc_input[:, health_inputs:])
            input_layers.append(misc_input_layer)
            layers_for_merge.append(misc_input_layer)
            inputs.append(misc_input[:, health_inputs:])

            layers_for_merge.append(network)
            network = ls.ConcatLayer(layers_for_merge)

        network = ls.DenseLayer(network, 512, nonlinearity=rectify,
                                W=weights_init, b=lasagne.init.Constant(0.1))

        network = ls.DenseLayer(network, output_size, nonlinearity=None, b=lasagne.init.Constant(.1))
        return network, input_layers, inputs


class OneHotHealthDQN(DQN):
    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]
        # weights_init = lasagne.init.GlorotUniform("relu")
        weights_init = lasagne.init.HeNormal("relu")

        network = ls.InputLayer(shape=img_input_shape, input_var=img_input)
        input_layers.append(network)
        network = ls.Conv2DLayer(network, num_filters=32, filter_size=8, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=4)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=4, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=2)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=3, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=1)
        network = ls.FlattenLayer(network)

        if self.misc_state_included:
            health_inputs = 4
            units_per_health_input = 100
            layers_for_merge = []
            for i in range(health_inputs):
                oh_input = lasagne.utils.one_hot(misc_input[:, i] - 1, units_per_health_input)
                health_input_layer = ls.InputLayer(shape=(None, units_per_health_input), input_var=oh_input)
                inputs.append(oh_input)
                input_layers.append(health_input_layer)
                layers_for_merge.append(health_input_layer)

            misc_input_layer = ls.InputLayer(shape=(None, misc_len - health_inputs),
                                             input_var=misc_input[:, health_inputs:])
            input_layers.append(misc_input_layer)
            layers_for_merge.append(misc_input_layer)
            inputs.append(misc_input[:, health_inputs:])

            layers_for_merge.append(network)
            network = ls.ConcatLayer(layers_for_merge)

        network = ls.DenseLayer(network, 512, nonlinearity=rectify,
                                W=weights_init, b=lasagne.init.Constant(0.1))

        network = ls.DenseLayer(network, output_size, nonlinearity=None, b=lasagne.init.Constant(.1))
        return network, input_layers, inputs


class OneHotAllDQN(DQN):
    def _initialize_network(self, img_input_shape, misc_len, output_size, img_input, misc_input=None, **kwargs):
        input_layers = []
        inputs = [img_input]
        # weights_init = lasagne.init.GlorotUniform("relu")
        weights_init = lasagne.init.HeNormal("relu")

        network = ls.InputLayer(shape=img_input_shape, input_var=img_input)
        input_layers.append(network)
        network = ls.Conv2DLayer(network, num_filters=32, filter_size=8, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=4)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=4, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=2)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=3, nonlinearity=rectify, W=weights_init,
                                 b=lasagne.init.Constant(0.1), stride=1)
        network = ls.FlattenLayer(network)

        if self.misc_state_included:
            layers_for_merge = []

            health_inputs = 4
            units_per_health_input = 100
            for i in range(health_inputs):
                oh_input = lasagne.utils.one_hot(misc_input[:, i] - 1, units_per_health_input)
                health_input_layer = ls.InputLayer(shape=(None, units_per_health_input), input_var=oh_input)
                inputs.append(oh_input)
                input_layers.append(health_input_layer)
                layers_for_merge.append(health_input_layer)

            time_inputs = 4
            # TODO set this somewhere else cause it depends on skiprate and timeout ....
            units_pertime_input = 525
            for i in range(health_inputs,health_inputs+time_inputs):
                oh_input = lasagne.utils.one_hot(misc_input[:, i] - 1, units_pertime_input)
                time_input_layer = ls.InputLayer(shape=(None, units_pertime_input), input_var=oh_input)
                inputs.append(oh_input)
                input_layers.append(time_input_layer)
                layers_for_merge.append(time_input_layer)

            other_misc_input = misc_input[:, health_inputs+time_inputs:]
            other_misc_shape = (None, misc_len - health_inputs-time_inputs)
            other_misc_input_layer = ls.InputLayer(shape=other_misc_shape,
                                             input_var=other_misc_input)
            input_layers.append(other_misc_input_layer)
            layers_for_merge.append(other_misc_input_layer)
            inputs.append(other_misc_input)

            layers_for_merge.append(network)
            network = ls.ConcatLayer(layers_for_merge)

        network = ls.DenseLayer(network, 512, nonlinearity=rectify,
                                W=weights_init, b=lasagne.init.Constant(0.1))

        network = ls.DenseLayer(network, output_size, nonlinearity=None, b=lasagne.init.Constant(.1))
        return network, input_layers, inputs
