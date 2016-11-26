class Perceptron(object):
    def __init__(self, input_num, activator):
        """
        Initialize perceptron
        :param input_num: the numbers of input
        :param activator: active function (double -> double)
        """
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        """
        print weights and bias after training
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """
        ZIP input_vec[x1, x2, x3,...] and weights[w1, w2, w3,...] to [(x1,w1),(x2,w2),(x3,w3),...]
        Use map function to calculate [x1*w1, x2*w2, x3*w3]
        Use reduce to get sum of the array
        """
        return self.activator(sum(map(lambda element: element[0] * element[1], list(zip(input_vec, self.weights))), self.bias))

    def train(self, input_vecs, labels, iteration, rate):
        """
        :param iteration: train times
        :param rate: study rate
        :return:
        """
        for index in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        """
        training per iteration
        """
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            # print self
            output = self.predict(input_vec)
            self._update_weight(input_vec, output, label, rate)

    def _update_weight(self, input_vec, output, label, rate):
        """
        Update weight by the rule of perceptron
        """
        delta = label - output
        self.weights = list(map(lambda element: element[1] + rate * delta * element[0], list(zip(input_vec, self.weights))))
        self.bias += rate * delta


def f(x):
    """
    Define active function
    """
    return 1 if x > 0 else 0


def get_training_dataset():
    """
    Create train data based on and Truth table
    """
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    # labels = [1, 0, 1, 1]
    return input_vecs, labels


def train_and_perceptron():
    """
    Train perceptron with and Truth table
    """
    p = Perceptron(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p


if __name__ == '__main__':
    and_perceptron = train_and_perceptron()
    print(and_perceptron)

    print('1 and 1 = %d' % and_perceptron.predict([1, 1]))
    print('0 and 0 = %d' % and_perceptron.predict([0, 0]))
    print('1 and 0 = %d' % and_perceptron.predict([1, 0]))
    print('0 and 1 = %d' % and_perceptron.predict([0, 1]))
