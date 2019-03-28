import random


class Connection:
    """
    Class with the weight and delta weights(used for momentum)
    """
    def __init__(self):
        self.weight = random.random()
        self.delta_weight = random.random()


class Neuron:
    """
    Neuron class
    """
    activation_function = None
    activation_function_derivative = None
    learning_rate = None
    momentum = None

    def __init__(self, num_output, index, activation_function, activation_function_derivative, learning_rate, momentum):
        """
        Constructor for neuron

        :param num_output: The number of neurons in the next layer
        :param index: The index/position of the neuron in its layer
        :param activation_function: The activation function
        :param activation_function_derivative: The derivative of the activation function
        :param learning_rate: The learning rate
        :param momentum: The momentum
        """

        self.gradient = 0
        self.output = random.random()

        self.output_weights = []
        for i in range(num_output):
            self.output_weights.append(Connection())

        self.index = index
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.learning_rate = learning_rate
        self.momentum = momentum

    def feed_forward(self, prev_layer):
        """
        Calculates the output of the neuron by summing the outputs of neurons in the previous layer and multiplying them with their weights and
        then put that sum in the activation function.

        :param prev_layer: The previous layer (list with neurons)
        :return: void
        """

        total = 0
        for prev_neuron in prev_layer:
            total += prev_neuron.output * prev_neuron.output_weights[self.index].weight

        self.output = self.activation_function(total)

    def calculate_output_gradient(self, target_value):
        """
        Calculate and set the gradient for an output neuron.

        :param target_value: The expected value
        :return: void
        """
        self.gradient = self.activation_function_derivative(self.output) * (target_value - self.output)

    def calculate_hidden_gradient(self, next_layer, has_bias):
        """
        Calculate and set the gradient for a neuron in a hidden layer

        :param next_layer: The next layer
        :param has_bias: if the neural network has a bias neuron
        :return: void
        """

        self.gradient = self.activation_function_derivative(self.output) * self.calculate_proportional_responsibility(next_layer, has_bias)

    def calculate_proportional_responsibility(self, next_layer, has_bias):
        """
        Calculate the proportional responsibility.

        :param next_layer: The next layer
        :param has_bias: if the neural network has a bias neuron
        :return: void
        """

        total = 0
        # For each neuron in next layer (excluding bias if there is one)
        for i in range(len(next_layer) - has_bias):
            neuron = next_layer[i]
            total += self.output_weights[neuron.index].weight * neuron.gradient

        return total

    def update_input_weights(self, prev_layer):
        """
        Updates the input weights based on the set gradients
        The neuron uses the momentum strategy

        :param prev_layer: The previous layer
        :return: void
        """

        # update all weights to this neuron in the previous layer
        # For each neuron in previous layer
        for neuron in prev_layer:
            old_delta_weight = neuron.output_weights[self.index].delta_weight

            new_delta_weight = (
                # Learning rate
                self.learning_rate *
                neuron.output *
                self.gradient +
                # Momentum
                old_delta_weight * self.momentum
            )

            neuron.output_weights[self.index].delta_weight = new_delta_weight
            neuron.output_weights[self.index].weight += new_delta_weight
