from functools import reduce


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        """
        Add the connection to the node of next layer
        """
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        """
        Calculate the delta of node in hidden layer
        """
        downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream,
                                  0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str
