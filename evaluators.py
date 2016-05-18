import lasagne
import lasagne.layers as ls
import numpy as np
import theano
import theano.tensor as tensor
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error


class DQN:
    def __init__(self, state_format, actions_number, architecture=None, gamma=0.99, learning_rate=0.00025,
                 freeze=False):
        self._inputs = dict()
        if architecture is None:
            architecture = dict()

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
        self._freeze = freeze

        network_image_input_shape = list(state_format["s_img"])
        network_image_input_shape.insert(0, None)

        # save it for the evaluation reshape
        self._single_image_input_shape = list(network_image_input_shape)
        self._single_image_input_shape[0] = 1

        architecture["img_input_shape"] = network_image_input_shape
        architecture["misc_len"] = self._misc_len
        architecture["output_size"] = actions_number

        self.network = self._initialize_network(**architecture)
        if self._freeze:
            self.frozen_network = self._initialize_network(**architecture)
        # print "Network initialized."
        self._learning_rate = learning_rate
        self._compile()

    def _initialize_network(self, img_input_shape, misc_len, output_size, **kwargs):

        network = ls.InputLayer(shape=img_input_shape, input_var=self._inputs["X"])

        network = ls.Conv2DLayer(network, num_filters=32, filter_size=8,
                                 nonlinearity=rectify, W=lasagne.init.GlorotUniform("relu"),
                                 b=lasagne.init.Constant(.1), stride=4, pad=2)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=4,
                                 nonlinearity=rectify, W=lasagne.init.GlorotUniform("relu"),
                                 b=lasagne.init.Constant(.1), stride=2)
        network = ls.Conv2DLayer(network, num_filters=64, filter_size=3,
                                 nonlinearity=rectify, W=lasagne.init.GlorotUniform("relu"),
                                 b=lasagne.init.Constant(.1), stride=1)

        if self._misc_state_included:
            network = ls.FlattenLayer(network)
            misc_input_layer = ls.InputLayer(shape=(None, misc_len), input_var=self._inputs["X_misc"])
            network = ls.ConcatLayer([network, misc_input_layer])

        network = ls.DenseLayer(network, 512, nonlinearity=rectify,
                                W=lasagne.init.GlorotUniform("relu"), b=lasagne.init.Constant(.1))

        network = ls.DenseLayer(network, output_size, nonlinearity=None, b=lasagne.init.Constant(.1))
        return network

    def _compile(self):

        q = ls.get_output(self.network, deterministic=False)
        deterministic_q = ls.get_output(self.network, deterministic=True)
        if self._freeze:
            frozen_q = ls.get_output(self.frozen_network, deterministic=True)

        a = self._inputs["A"]
        r = self._inputs["R"]
        nonterminal = self._inputs["Nonterminal"]
        q2 = self._inputs["Q2"]

        # TODO move r + ... out and check ig it's faster
        # target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + self._gamma * nonterminal * q2)

        loss = squared_error(q[tensor.arange(q.shape[0]), a], r + self._gamma * nonterminal * q2).mean()

        params = ls.get_all_params(self.network, trainable=True)
        # TODO enable learning_rate changing after compilation
        updates = lasagne.updates.rmsprop(loss, params, self._learning_rate)

        # print "Compiling Theano functions ..."

        # TODO find out why this causes problems with misc vector
        # mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        mode = None
        if self._misc_state_included:
            self._learn = theano.function([self._inputs["X"], self._inputs["X_misc"], q2, a, r, nonterminal], loss,
                                          updates=updates, mode=mode, name="learn_fn")
            self._evaluate = theano.function([self._inputs["X"], self._inputs["X_misc"]], deterministic_q, mode=mode,
                                             name="eval_fn")
            if self._freeze:
                self._q2_evaluate = theano.function([self._inputs["X"], self._inputs["X_misc"]], frozen_q, mode=mode,
                                                    name="frozen_eval_fn")
            else:
                self._q2_evaluate = self._evaluate
        else:
            self._learn = theano.function([self._inputs["X"], q2, a, r, nonterminal], loss, updates=updates, mode=mode,
                                          name="learn_fn")
            self._evaluate = theano.function([self._inputs["X"]], deterministic_q, mode=mode, name="eval_fn")
            if self._freeze:
                self._q2_evaluate = theano.function([self._inputs["X"]], frozen_q, mode=mode, name="frozen_eval_fn")
            else:
                self._q2_evaluate = self._evaluate

    def learn(self, transitions):
        # Learning approximation: Q(s1,t+1) = r + nonterminal *Q(s2,t)
        X = transitions["s1_img"]
        X2 = transitions["s2_img"]
        if self._misc_state_included:
            X_misc = transitions["s1_misc"]
            X2_misc = transitions["s2_misc"]
            Q2 = np.max(self._q2_evaluate(X2, X2_misc), axis=1)
        else:
            Q2 = np.max(self._q2_evaluate(X2), axis=1)

        if self._misc_state_included:
            loss = self._learn(X, X_misc, Q2, transitions["a"], transitions["r"], transitions["nonterminal"])
        else:
            loss = self._learn(X, Q2, transitions["a"], transitions["r"], transitions["nonterminal"])

        self._loss_history.append(loss)

    def estimate_best_action(self, state):
        if self._misc_state_included:
            qvals = self._evaluate(state[0].reshape(self._single_image_input_shape),
                                   state[1].reshape(1, self._misc_len))
            # TODO Check if it's correct
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
        return self.network

    def melt(self):
        ls.set_all_param_values(self.frozen_network, ls.get_all_param_values(self.network))


class CNNEvaluator(DQN):
    # TODO check if this constructor is needed
    def __init__(self, learning_rate=0.00001, gamma=1.0, **kwargs):
        DQN.__init__(self, learning_rate, gamma, **kwargs)

    def _initialize_network(self, img_input_shape, misc_len, output_size, conv_layers=3, num_filters=(32, 32, 32),
                            filter_size=(7, 5, 3), hidden_units=(1024), pool_size=(2, 2, 2), stride=(1, 1, 1),
                            pad=(0, 0, 0),
                            hidden_layers=1, dropout=False):

        network = ls.InputLayer(shape=img_input_shape, input_var=self._inputs["X"])

        for i in range(conv_layers):
            network = ls.Conv2DLayer(network, num_filters=num_filters[i], filter_size=filter_size[i],
                                     nonlinearity=rectify, W=lasagne.init.GlorotUniform("relu"),
                                     b=lasagne.init.Constant(.1), stride=stride[i], pad=pad[i])
            if pool_size is not None:
                network = ls.MaxPool2DLayer(network, pool_size=pool_size[i])
            if dropout:
                network = lasagne.layers.dropout(network, p=0.5)

        if self._misc_state_included:
            network = ls.FlattenLayer(network)
            misc_input_layer = ls.InputLayer(shape=(None, misc_len), input_var=self._inputs["X_misc"])
            network = ls.ConcatLayer([network, misc_input_layer])

        for i in range(hidden_layers):
            network = ls.DenseLayer(network, hidden_units[i], nonlinearity=rectify,
                                    W=lasagne.init.GlorotUniform("relu"), b=lasagne.init.Constant(.1))
            if dropout:
                network = lasagne.layers.dropout(network, p=0.5)
        network = ls.DenseLayer(network, output_size, nonlinearity=None, b=lasagne.init.Constant(.1))
        return network


class CNNEvaluatorMem(DQN):
    def __init__(self, **kwargs):
        DQN.__init__(self, **kwargs)

    def _initialize_network(self, img_input_shape, misc_len, output_size, conv_layers=3, num_filters=(32, 32, 32),
                            filter_size=((5, 5), (5, 5), (5, 5)), hidden_units=(1024),
                            pool_size=((2, 2), (2, 2), (2, 2)),
                            hidden_layers=1, dropout=False, memory=1, merge_hidden=(512)):

        memory = max(1, memory)
        channels_per_cell = img_input_shape[1] / memory
        networks = []
        shape = img_input_shape[0:4]
        shape[1] /= memory

        for mem in range(memory):
            start_i = mem * channels_per_cell
            end_i = start_i + channels_per_cell
            cell_input = ls.InputLayer(shape=shape, input_var=self._inputs["X"][:, start_i:end_i])
            networks.append(cell_input)

        for i in range(conv_layers):
            for mem in range(memory):
                if mem == 0:
                    w = lasagne.init.GlorotUniform("relu")
                    b = lasagne.init.Constant(.1)
                else:
                    w = networks[mem - 1].W
                    b = networks[mem - 1].b
                networks[mem] = ls.Conv2DLayer(networks[mem], num_filters=num_filters[i], filter_size=filter_size[i],
                                               nonlinearity=rectify, W=w, b=b)

            if dropout or pool_size is not None:
                for mem in range(memory):
                    if pool_size is not None:
                        networks[mem] = ls.MaxPool2DLayer(networks[mem], pool_size=pool_size[i])
                    if dropout:
                        networks[mem] = lasagne.layers.dropout(networks[mem], p=0.5)

        for mem in range(memory):
            if mem == 0:
                w = lasagne.init.GlorotUniform("relu")
                b = lasagne.init.Constant(0.1)
            else:
                w = networks[mem - 1].W
                b = networks[mem - 1].b
            networks[mem] = ls.FlattenLayer(networks[mem])
            networks[mem] = ls.DenseLayer(networks[mem], merge_hidden, nonlinearity=rectify,
                                          W=w, b=b)

        if self._misc_state_included:
            misc_input_layer = ls.InputLayer(shape=(None, misc_len), input_var=self._inputs["X_misc"])
            networks.append(misc_input_layer)
            network = ls.ConcatLayer(networks)
        else:
            network = ls.ConcatLayer(networks)

        for i in range(hidden_layers):
            network = ls.DenseLayer(network, hidden_units[i], nonlinearity=rectify,
                                    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.1))
            if dropout:
                network = lasagne.layers.dropout(network, p=0.5)

        network = ls.DenseLayer(network, output_size, nonlinearity=None)
        return network
