import math
import random
from neuron import Neuron


class HyperbolicTangent:
    @staticmethod
    def activation_function(x):
        return math.tanh(x)

    @staticmethod
    def activation_function_derivative(x):
        # TODO: Ask Huib
        # return 1 - math.tanh(math.tanh(x))
        return 1 - x * x


class Rectifier:
    @staticmethod
    def activation_function(x):
        return max(0, x)

    @staticmethod
    def activation_function_derivative(x):
        # NOTE: https://www.quora.com/How-do-we-compute-the-gradient-of-a-ReLU-for-backpropagation
        return x > 0


class Step:
    @staticmethod
    def activation_function(x):
        return 1 if x >= 0.5 else 0

    @staticmethod
    def activation_function_derivative(x):
        assert "No activation derivative provided"


class Net:
    has_bias = None

    def __init__(self, topology, activation_strategy, learning_rate = 0.1, momentum = 0.5, has_bias = True):
        """
        Constructor for the Neural network

        :param topology:
        The topology of the network in form of an list with the number of neurons for each layer:
        [1, 2, 1] -> 1 input neuron, 1 hidden layer with 2 neurons and 1 output neuron
        :param activation_strategy: A class with an activation function and its derivative function
        :param learning_rate: The learning speed [0 ... 1]
        :param momentum: The momentum of the network [0 ... 1] (https://www.quora.com/What-does-momentum-mean-in-neural-networks)
        :param has_bias: If the network should create bias neurons
        """
        self.layers = []
        self.has_bias = has_bias

        print("Created net")
        for i in range(len(topology)):
            layer = []
            num_neurons = topology[i]
            num_output_weights = 0 if i == len(topology) - 1 else topology[i + 1]

            # Create all neurons + bias (if enabled)
            total_neurons = num_neurons + self.has_bias
            for neuron_index in range(total_neurons):
                layer.append(
                    Neuron(
                        num_output_weights,
                        neuron_index,
                        activation_strategy.activation_function,
                        activation_strategy.activation_function_derivative,
                        learning_rate,
                        momentum
                    )
                )
                print("Created " + "Neuron" if not neuron_index == total_neurons - 1 else "bias Neuron")
                if neuron_index == total_neurons:
                    layer[-1].output = 1

            # Add layer to the net
            self.layers.append(layer)
            print()

    def back_propagate(self, target_values):
        """
        Back propagation function for the network.
        The network uses the values from the last time the feed forward function was used.

        :param target_values: The expected output value with the current input
        :return: void
        """
        # For each neuron (Excluding the bias) in output calculate the gradient
        output_layer = self.layers[-1]
        for i in range(len(output_layer) - self.has_bias):
            neuron = output_layer[i]
            neuron.calculate_output_gradient(target_values[i])

        # For each other layer, from back to front calculate the gradient
        for i in reversed(range(len(self.layers) - 1)):
            cur_layer = self.layers[i]
            next_layer = self.layers[i+1]

            # For each neuron in that layer
            for neuron in cur_layer:
                neuron.calculate_hidden_gradient(next_layer, self.has_bias)

        # Now that we have all gradients we can calculate the new weights
        for i in range(1, len(self.layers)):
            cur_layer = self.layers[i]
            prev_layer = self.layers[i - 1]

            # for each neuron in current layer (excluding bias)
            for j in range(len(cur_layer) - self.has_bias):
                neuron = cur_layer[j]
                neuron.update_input_weights(prev_layer)

    def feed_forward(self, input_values):
        """
        The feed forward function for the network.

        :param input_values: The values to feed forward
        :return: void
        """
        if len(input_values) != len(self.layers[0]) - self.has_bias:  # Excluding bias
            raise Exception("Given input does not match topology")

        # Set input values
        for i in range(len(input_values)):
            # Neuron i in first layer
            self.layers[0][i].output = input_values[i]

        # Forward propagation from second layer to output
        for i in range(1, len(self.layers)):
            cur_layer = self.layers[i]
            prev_layer = self.layers[i - 1]

            for j in range(len(cur_layer) - self.has_bias):  # Excluding bias
                cur_layer[j].feed_forward(prev_layer)

    def get_results(self):
        """
        Returns the current output.
        The network uses the values from the last time the feed forward function was used.

        :return: A list with the values of all output neurons
        """
        return [self.layers[-1][x].output for x in range(len(self.layers[-1]) - self.has_bias)]  # Exclude bias

    def __repr__(self):
        """
        Print function for the network

        :return: The string to print
        """
        max_layer_size = 0
        for layer in self.layers:
            max_layer_size = max(max_layer_size, len(layer))

        return_str = ""
        for layer in self.layers:
            diff = int((max_layer_size - len(layer)) / 2)

            for _ in range(diff):
                return_str += '\t'

            for neuron in layer:
                return_str += str(neuron.index) + '(' + str(round(neuron.output)) + " " + str([round(x.weight, ndigits=2) for x in neuron.output_weights]) + ')' + '\t'

            for _ in range(diff):
                return_str += '\t'
            return_str += '\n\n'
        return return_str[:-2]  # Remove trailing '\n\n'

# Debug functionality - Basic XOR
if __name__ == "__main__":
    # For some reason the HyperbolicTangent does not work so we use the Rectifier
    xor_gate = Net([2, 2, 1], HyperbolicTangent)

    # Set weights for testing
    layers = xor_gate.layers
    layers[0][0].output_weights[0].weight = 0.2
    layers[0][0].output_weights[1].weight = 0.7

    layers[0][1].output_weights[0].weight = -0.4
    layers[0][1].output_weights[1].weight =  0.1

    layers[1][0].output_weights[0].weight = 0.6
    layers[1][1].output_weights[0].weight = 0.9
    print(xor_gate)
    print()
    print()

    xor_gate.feed_forward([1, 1])
    print("Output:")
    print(xor_gate.get_results())
    print("Target:")
    print([0])
    xor_gate.back_propagate([0])

    print()
    print()
    print(xor_gate)

    xor_gate.feed_forward([1, 0])
    print("Output:")
    print(xor_gate.get_results())
    print("Target:")
    print([1])
    xor_gate.back_propagate([1])

    print()
    print()
    print(xor_gate)

    if False:
        exit()

    training_set = [
        {"inp": [1, 0], "outp": [1]},
        {"inp": [1, 1], "outp": [0]},
        {"inp": [0, 1], "outp": [1]},
        {"inp": [0, 0], "outp": [0]},
    ]

    count = 0
    for ind in range(10000):
        cur = training_set[ind % 4]

        print("Input:")
        print(cur["inp"])
        xor_gate.feed_forward(cur["inp"])

        print("Output:")
        print(xor_gate.get_results())
        print("Target:")
        print(cur["outp"])

        # Train
        xor_gate.back_propagate(cur["outp"])

        # If result is correct times x times in a row we break
        # This could be luck, so we should not do this outside of testing
        if (xor_gate.get_results()[0] > 0.5) == cur["outp"][0]:
            count += 1
        else:
            count = 0

        if count > 100:
            break

        print()

    print("Success: ", count)

    # Expected: False, True, True, False
    print(xor_gate)
    xor_gate.feed_forward([1, 1])
    print(xor_gate.get_results()[0] > 0.5, xor_gate.get_results()[0])

    xor_gate.feed_forward([1, 0])
    print(xor_gate.get_results()[0] > 0.5, xor_gate.get_results()[0])

    xor_gate.feed_forward([0, 1])
    print(xor_gate.get_results()[0] > 0.5, xor_gate.get_results()[0])

    xor_gate.feed_forward([0, 0])
    print(xor_gate.get_results()[0] > 0.5, xor_gate.get_results()[0])
