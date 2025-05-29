"""'
    THIS IMPLEMENTATION IS NOY ANY WAY NEAR TO efficient or optimal , it's just for educational purposes that focuses mainly on
    1. deep understanding of NN
    2. implementation skill

    ########################################################################################
     in this file we will implement NN in iterative manner (most of the code anyway)
     naming convention

    net = W*input to the layer
    out = activation(net)

    I use them loosely without mentioning the index of the layer but this is almost 100% iterative solution so ther is no need for sepecific indexing
    I can see why it's important in a vectorized implementation setting
"""

import numpy as np
from functools import partial
from typing import List, Optional
from collections import namedtuple


class Neuron:
    """
    from feed forward we have
    x -> net = w*x : out = activation(net) .... net = w*out: out= activation(net)
    thus we will cache inputs for each layer
    since in backprop we have
    dE/dnet = dE/dout(for the current layer) * dout/dnet
    and dince dout = activation(net) then
    dout/dnet = dactivation/dnet
    and net is just w*input for this layer
    """

    def __init__(self, activation: str, weights, **kwargs):
        self.activation = activation.lower()

        self.kwargs = kwargs
        self.weights = weights
        self.make_activations_dictioanry()
        self.make_activations_der_dictionary()

        self.validate_activation()

    def validate_activation(self):
        """
            sanity check for valid activation names that are supported in this class which are
            1. Sigmoid
            2. Polynomial
            3. Identity
            4.Tanh

        :raise: ValueError if the passed activation function to the initializer is not supported
        """
        if self.activation not in self.activations.keys():
            raise ValueError(" un supported activation function")

    def make_activations_dictioanry(self):
        """
        this function will make a dictionary of activation functions that are build in this class
        :return: dictionary{'function_name':function}
        """
        sigmoid = partial(self.sigmoid, **self.kwargs)
        tanh = partial(self.tanh, **self.kwargs)
        identity = partial(self.identity, **self.kwargs)
        polynomial = partial(self.polynomial, **self.kwargs)

        self.activations = {"sigmoid": sigmoid, "tanh": tanh, "identity": identity, "polynomial": polynomial}

    def make_activations_der_dictionary(self):
        """
        this function will make a dictionary of derivatives of activation functions that are build in this class
        :return: dictionary{'function_name':function_derivative}
        """
        sigmoid = partial(self.sigmoid_der, **self.kwargs)
        tanh = partial(self.tanh_der, **self.kwargs)
        identity = partial(self.identity_der, **self.kwargs)
        polynomial = partial(self.polynomial_der, **self.kwargs)

        self.activations_der = {"sigmoid": sigmoid, "tanh": tanh, "identity": identity, "polynomial": polynomial}

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def sigmoid_der(self):
        input = self.net(self.inputs)
        return self.sigmoid(input) * (1 - self.sigmoid(input))

    def tanh(self, input):
        return np.tanh(input)

    def tanh_der(self):
        input = self.net(self.inputs)
        return 1 - (self.tanh(input)) ** 2

    def identity(self, input):
        return input

    def identity_der(self):
        input = self.net(self.inputs)
        return input

    def polynomial(self, input, degree):
        return np.pow(input, degree)

    def polynomial_der(self, degree):
        """
        derivative of any polynomial X^n == n*x^(n-1)
        :param degree: degree of the polynomial
        :return:
        """
        input = self.net(self.inputs)
        return degree * np.pow(input, degree - 1)

    def net(self, inputs):
        """
        useful utility for feed_forward
        :param inputs:
        :return:
        """
        mul = lambda x, y: x * y
        return sum(mul(input, weight) for input, weight in zip(inputs, self.weights))

    def output(self, input):
        """
        apply the activation function over the input
        :param input:
        :return: activation(input)
        """

        # simple sanity check that length of the weight  == length of the given input
        if len(self.weights) != len(input):
            raise ValueError("length of weights must = length of the inputs")

        self.inputs = input  # we don't need this , but you may use it in debugging
        return self.activations[self.activation](self.net(input))

    def dnet_dout(self):
        """
        out = activation(net)
        dout = activation_der(net)  which is what the function doing
        :return:  derivative of the activation function
        """
        return self.activations_der[self.activation]()

    # i used it for debugging
    def __str__(self):
        return f"{self.weights=} , {self.activation=}"


class NeuralLayer:

    def __init__(
        self,
        size,  # how many neurons in the layer
        activation,  # which of activation function to use
        weights,  # initial weight for the layer
        **kwargs,  # any kwargs to be passed for Neuron like degree of polynomial if it's used !
    ):
        self.size = size
        self.weights = weights

        self.activation = activation
        self.kwargs = kwargs

        self.neurons = [Neuron(activation, weights, **kwargs) for weights in self.weights]

    def feed_forward(self, input):
        """
        mitigate the input through the network
        :param input:
        :return: the final prediction
        """
        self.input = input
        self.output = [neuron.output(input) for neuron in self.neurons]
        return self.output

    def dnet_dout(self):
        """
        calculate  dnet/dout  by passing the call to each neuron
        :return: [ dnet/dout ]
        """
        return [neuron.dnet_dout() for neuron in self.neurons]


class NeuralNetwork:
    def __init__(
        self,
        layers: List[int],  # size of each layer in the network
        weights,  # initial weights for the whole network
        activations,  # type of activation for  each layer
        lr,  # learning rate
        **kwargs,  # any kwargs to be passed for other constructions
    ):

        # sanity check for layers and activations
        self.validate_layers(layers, activations)
        self.validate_activations(activations)

        # make layers
        self.weights = weights
        self.layers = self.make_layers(layers, activations, kwargs=kwargs)
        self.lr = lr

    def feed_forward(self, input):
        self.input = input  # cache it for backpropagation

        for layer in self.layers:
            output = layer.feed_forward(input)
            input = output

        return output

    def derror_dout(self, output):
        """
             this function will calculate derror/douput
             assuming we are using simple mean squared error 1/2(predicted-output)^2
        :return: the derivative (predicted - output)
        """

        predicted = self.feed_forward(self.input)

        return [prediction - output for prediction, output in zip(predicted, output)]

    def cache_delta(self, output):
        """
        this function will do a  backward step to cache all derror/dout and derror/dnet for each layer  to be used later
        in updating weights
        Note: we don't actually need the derror/dout for each layer but i did it for debugging purposes
        """
        Layer = namedtuple("Layer", "dout dnet")
        self.deltas = []

        # iterate over layers from the end
        prev = self.derror_dout(output)

        for layer in reversed(self.layers):  # start form the last layer
            dout = prev
            dout_dnet = layer.dnet_dout()
            derror_dnet = [net * out for net, out in zip(dout, dout_dnet)]  # derror/dout * dout/dnet
            self.deltas.append(Layer(dout, derror_dnet))

            # move to the prev layer
            # note that net=( dout(prev) *weights)
            prev = layer.weights.T @ np.array(derror_dnet).reshape(-1, 1)

        for delta in self.deltas:
            print(delta)

    def update_weights(self):
        """
        Note that if we are in layer i then Net = W*output_{i-1} (output form the previous layer)
        thus derror/dw = derror/dnet (which we cached) * output{i-1}
        we can start form the beginning if we want
        - i am a bit lazy for this one , so i will go with matrices
        :return:
        """

        weights = []
        for index, (layer, d, weight) in enumerate(zip(reversed(self.layers), self.deltas, reversed(self.weights))):
            #  input of this layer == output from the previous layer!
            # reshaping for matrix multiplication

            dnet = np.array(d.dnet).reshape(-1, 1)

            output = np.array(layer.input).reshape(-1, 1)

            weight -= self.lr * dnet @ output.T

            weights.append(weight)
        print(weights)

    def train_step(self, input, output):
        self.feed_forward(input)
        self.cache_delta(output)
        self.update_weights()

    def make_layers(self, layers, activations, kwargs):
        # if the user passed a singed activation : this means ,using it for the whole network
        if type(activations) == str:
            activations = [activations] * len(layers)

        return [
            NeuralLayer(layer, activation, weight, **kwargs)
            for layer, activation, weight in zip(layers, activations, self.weights)
        ]

    def validate_layers(self, layers, activations):
        # check all are int
        if type(layers) != list:
            raise TypeError(" layer sizes must be a list of int")

        # some duplicate code !
        for layer in layers:
            if type(layer) != int:
                raise TypeError("Layer size must be int ")

        # length is the same as activation if it's a list
        if type(activations) == list and len(activations) != len(layers):
            raise ValueError("activations must = layers")

    def validate_activations(self, activations):
        # check that all are string  if it's a list
        if type(activations) in (list, str):
            if type(activations) == list:
                for activation in activations:
                    if type(activation) != str:
                        raise TypeError("Activation  function must be string")
        else:
            raise TypeError("Activation must be either a list of string or string")


if __name__ == "__main__":

    # test 1
    # net = NeuralNetwork([2, 2], weights, activations="polynomial", lr=0.5, degree=2)
    # input = [1, 1]
    # net.train_step(input, output)

    weights = [
        np.array([[0.1, 0.1], [0.2, 0.1], [0.1, 0.3], [0.5, 0.01]], dtype=np.float64),
        np.array([[0.1, 0.2, 0.1, 0.2], [0.1, 0.1, 0.1, 0.5], [0.1, 0.4, 0.3, 0.2]], dtype=np.float64),
    ]
    output = np.array([0.4, 0.7, 0.6])
    net = NeuralNetwork([2, 4, 3], weights, activations="sigmoid", lr=0.5)
    input = [1, 2]
    net.train_step(input, output)
