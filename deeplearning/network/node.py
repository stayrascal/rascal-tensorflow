from functools import reduce


class Node(object):
    def __init__(self, layer_index, node_index):
        """
        Create node object
        """
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        """
        Set the output of node, will use this function if the node come from input layer
        """
        self.output = output

    def append_downstream_connection(self, conn):
        """
        Add the connection to the node of next layer
        """
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        """
        Add the connection to next of previous layer
        """
        self.upstream.append(conn)

    def calc_output(self):
        """
        Calculate the output of the node
        """
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        """
        Calculate the delta of node in hidden layer
        """
        downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream,
                                  0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        """
        Calculate the delta of noe in output layer
        """
        self.data = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str
