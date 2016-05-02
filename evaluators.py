import lasagne
import lasagne.layers as ls
import numpy as np
import theano
import theano.tensor as tensor
from lasagne.nonlinearities import tanh, rectify, leaky_rectify
from lasagne.objectives import squared_error
from lasagne.updates import sgd


def relu_weights_initializer(alpha=0.01):
    return lasagne.init.GlorotNormal(gain=np.sqrt(2 / (1 + alpha ** 2)))


class MLPEvaluator:
    def __init__(self, state_format, actions_number, architecture=dict(), gamma=0.99,
                 updates=sgd, learning_rate=0.01):

        self._inputs = dict()

        self._loss_history = []
        self._misc_state_included = (state_format["s_misc"] > 0)
        self._gamma = np.float64(gamma)
        if self._misc_state_included:
            self._inputs["X_misc"] = tensor.matrix("X_misc")
            self._misc_len = state_format["s_misc"]
        else:
            self._misc_len = None

        self._inputs["X"] = tensor.tensor4("X")
        self._inputs["Q2"] = tensor.vector("Q2")
        self._inputs["A"] = tensor.vector("Action", dtype="int32")
        self._inputs["R"] = tensor.vector("Reward")
        self._inputs["Nonterminal"] = tensor.vector("Nonterminal", dtype="int8")

        network_image_input_shape = list(state_format["s_img"])
        network_image_input_shape.insert(0, None)

        # save it for the evaluation reshape
        self._single_image_input_shape = list(network_image_input_shape)
        self._single_image_input_shape[0] = 1

        architecture["img_input_shape"] = network_image_input_shape
        architecture["misc_len"] = self._misc_len
        architecture["output_size"] = actions_number

        self._initialize_network(**architecture)
        # print "Network initialized."
        self._compile(updates, learning_rate)

    def _initialize_network(self, img_input_shape, misc_len, output_size, hidden_units=[500], hidden_layers=1,
                            hidden_nonlin=leaky_rectify, output_nonlin=tanh, updates=sgd):
        # print "Initializing MLP network..."
        # image input layer
        network = ls.InputLayer(shape=img_input_shape, input_var=self._inputs["X"])
        # hidden layers
        for i in range(hidden_layers):
            network = ls.DenseLayer(network, hidden_units[i], nonlinearity=hidden_nonlin, W=weights_initializer())

        # misc layer and merge with rest of the network
        if self._misc_state_included:
            # misc input layer
            misc_input_layer = ls.InputLayer(shape=(None, misc_len), input_var=self._inputs["X_misc"])
            # merge layer
            network = ls.ConcatLayer([network, misc_input_layer])

        # output layer
        network = ls.DenseLayer(network, output_size, nonlinearity=output_nonlin)
        self._network = network

    def _compile(self, updates, learning_rate):

        q = ls.get_output(self._network, deterministic=False)
        deterministic_q = ls.get_output(self._network, deterministic=True)

        a = self._inputs["A"]
        r = self._inputs["R"]
        nonterminal = self._inputs["Nonterminal"]
        q2 = self._inputs["Q2"]

        target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + self._gamma * nonterminal * q2)

        loss = squared_error(q, target_q).mean()
        regularized_loss = loss
        params = ls.get_all_params(self._network, trainable=True)
        # TODO enable learning_rate change after compilation
        updates = updates(regularized_loss, params, learning_rate)

        # print "Compiling Theano functions ..."

        # TODO find out why this causes problems with misc vector
        # mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        mode = None
        if self._misc_state_included:
            self._learn = theano.function([self._inputs["X"], self._inputs["X_misc"], q2, a, r, nonterminal], loss,
                                          updates=updates, mode=mode, name="learn_fn")
            self._evaluate = theano.function([self._inputs["X"], self._inputs["X_misc"]], deterministic_q, mode=mode,
                                             name="eval_fn")
        else:
            self._learn = theano.function([self._inputs["X"], q2, a, r, nonterminal], loss, updates=updates, mode=mode,
                                          name="learn_fn")
            self._evaluate = theano.function([self._inputs["X"]], deterministic_q, mode=mode, name="eval_fn")
            # print "Theano functions compiled."

    def learn(self, transitions):
        # Learning approximation: Q(s1,t+1) = r + nonterminal *Q(s2,t) 
        X = transitions["s1_img"]
        X2 = transitions["s2_img"]
        if self._misc_state_included:
            X_misc = transitions["s1_misc"]
            X2_misc = transitions["s2_misc"]
            Q2 = np.max(self._evaluate(X2, X2_misc), axis=1)
        else:
            Q2 = np.max(self._evaluate(X2), axis=1)
            # DEBUG
        # print Q2
        if self._misc_state_included:
            loss = self._learn(X, X_misc, Q2, transitions["a"], transitions["r"], transitions["nonterminal"])
        else:
            loss = self._learn(X, Q2, transitions["a"], transitions["r"], transitions["nonterminal"])

        self._loss_history.append(loss)

    def best_action(self, state):
        if self._misc_state_included:
            qvals = self._evaluate(state[0].reshape(self._single_image_input_shape),
                                   state[1].reshape(1, self._misc_len))
            a = np.argmax(qvals)
            # DEBUG
            # print np.max(qvals)
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
        return self._network


class CNNEvaluator(MLPEvaluator):
    def __init__(self, **kwargs):
        MLPEvaluator.__init__(self, **kwargs)

    def _initialize_network(self, img_input_shape, misc_len, output_size, conv_layers=2, num_filters=[32, 32],
                            filter_size=[(5, 5), (5, 5)], hidden_units=[256], pool_size=[(2, 2), (2, 2)],
                            hidden_layers=1, conv_nonlin=rectify,
                            hidden_nonlin=leaky_rectify, output_nonlin=None, dropout=False):

        # print "Initializing CNN ..."
        # image input layer
        network = ls.InputLayer(shape=img_input_shape, input_var=self._inputs["X"])

        # convolution and pooling layers
        for i in range(conv_layers):
            network = ls.Conv2DLayer(network, num_filters=num_filters[i], filter_size=filter_size[i],
                                     nonlinearity=conv_nonlin, W=relu_weights_initializer())
            network = ls.MaxPool2DLayer(network, pool_size=pool_size[i])
            if dropout:
                network = lasagne.layers.dropout(network, p=0.5)

        network = ls.FlattenLayer(network)

        if self._misc_state_included:
            # misc input layer
            misc_input_layer = ls.InputLayer(shape=(None, misc_len), input_var=self._inputs["X_misc"])
            # merge layer
            network = ls.ConcatLayer([network, misc_input_layer])

            # dense layers
        for i in range(hidden_layers):
            network = ls.DenseLayer(network, hidden_units[i], nonlinearity=hidden_nonlin, W=relu_weights_initializer())
            if dropout:
                network = lasagne.layers.dropout(network, p=0.5)
        # output layer
        network = ls.DenseLayer(network, output_size, nonlinearity=output_nonlin)
        self._network = network


class LinearEvaluator(MLPEvaluator):
    def __init__(self, **kwargs):
        MLPEvaluator.__init__(self, **kwargs)

    def _initialize_network(self, img_input_shape, misc_len, output_size, output_nonlin=None):
        # print "Initializing Linear evaluator ..."
        # image input layer
        network = ls.InputLayer(shape=img_input_shape, input_var=self._inputs["X"])

        if self._misc_state_included:
            # misc input layer
            misc_input_layer = ls.InputLayer(shape=(None, misc_len), input_var=self._inputs["X_misc"])
            # merge layer
            network = ls.ConcatLayer([network, misc_input_layer])

        # output layer
        network = ls.DenseLayer(network, output_size, nonlinearity=output_nonlin)
        self._network = network
