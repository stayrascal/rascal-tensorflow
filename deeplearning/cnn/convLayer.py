import numpy as np

from deeplearning.cnn.filter import Filter
from deeplearning.cnn.reluActivator import ReluActivator


def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


def conv(input_array, kernel_array, output_array, stride, bias):
    channel_numbber = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (get_patch(input_array, i, j, kernel_width, kernel_height, stride) * kernel_array).sum() + bias


def padding(input_array, zp):
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((input_depth, input_height + 2 * zp, input_width + 2 * zp))
            padded_array[:, zp: zp + input_height, zp: zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((input_height + 2 * zp, input_width + 2 * zp))
            padded_array[zp: zp + input_height, zp: zp + input_width] = input_array
            return padded_array


class ConvLayer(object):
    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, filter_number, zero_padding, stride, activator):
        self.activator = activator
        self.stride = stride
        self.zero_padding = zero_padding
        self.filter_number = filter_number
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.channel_number = channel_number
        self.input_height = input_height
        self.input_width = input_width
        self.output_width = ConvLayer.calculate_output_size(self.input_width, filter_width, zero_padding, stride)
        self.output_height = ConvLayer.calculate_output_size(self.input_height, filter_height, zero_padding, stride)
        self.output_array = np.zeros((self.filter_number, self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width, filter_height, self.channel_number))

    def forward(self, input_array):
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(), self.output_array[f], self.stride, filter.get_bias())
            element_wise_op(self.output_array, self.activator.forward)

    def bp_sensitivity_map(self, sensitivity_array, activator):
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)
        self.delta_array = self.create_delta_array()
        for f in range(self.filter_number):
            filter = self.filters[f]
            flipped_weights = np.array(map(lambda i: np.rot90(i, 2), filter.get_weights()))
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        expanded_array = np.zeros((depth, expanded_height, expanded_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expanded_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
        return expanded_array

    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    def bp_gradient(self, sensitivity_array):
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d], expanded_array[f], filter.weights_grad[d], 1, 0)
                filter.bias_grad = expanded_array[f].sum()

    def update(self):
        for filter in self.filters:
            filter.update(self.learning_rate)

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size - filter_size + 2 * zero_padding) / stride + 1


def init_test():
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])
    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],
         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]])
    cl = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, ReluActivator(), 0.001)
    cl.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],
         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],
         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)
    cl.filters[0].bias = 1
    cl.filters[1].weights = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],
         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],
         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)
    return a, b, cl


def gradient_check():
    error_function = lambda o: o.sum()

    a, b, cl = init_test()
    cl.forward(a)

    sensitivity_array = np.ones(cl.output_array.shape, dtype=np.float64)

    cl.backward(a, sensitivity_array, ReluActivator())
    epsilon = 10e-4
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d, i, j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weights[d, i, j] -= 2 * epsilon
                cl.forward(a)
                err2 = error_function(cl.output_array)
                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[d, i, j] += epsilon
                print('weights(%d,%d,%d): expected - actural %f - %f' % (d, i, j, expect_grad, cl.filters[0].weights_grad[d, i, j]))

if __name__ == '__main__':
    gradient_check()
