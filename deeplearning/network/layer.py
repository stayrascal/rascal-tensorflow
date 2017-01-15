from deeplearning.network.constNode import ConstNode
from deeplearning.network.node import Node


class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for index in range(node_count):
            self.nodes.append(Node(layer_index, index))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        for index in range(len(data)):
            self.nodes[index].set_output(data[index])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)
