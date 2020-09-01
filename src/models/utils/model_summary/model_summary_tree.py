"""Module statistics tree."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import queue

LOGGER = logging.getLogger(__name__)


class ModelSummaryTree():
    """Module statistics tree."""

    def __init__(self, root_node):
        assert isinstance(root_node, ModelSummaryNode)
        self.root_node = root_node

    def get_same_level_max_node_depth(self, query_node):
        """Get maximum depth among nodes of the same level."""
        if query_node.name == self.root_node.name:
            same_level_depth = 0
        else:
            same_level_depth = max([child.depth for child in
                                    query_node.parent.children])
        return same_level_depth

    def update_summary_nodes_granularity(self):
        """Update the max depth among nodes of the same level."""
        node_queue = queue.Queue()
        node_queue.put(self.root_node)
        while not node_queue.empty():
            node = node_queue.get()
            node.granularity = self.get_same_level_max_node_depth(node)
            for child in node.children:
                node_queue.put(child)

    def get_collected_summary_nodes(self, query_granularity):
        """Get collected nodes given a query_granularity."""
        self.update_summary_nodes_granularity()

        collected_nodes = list()
        stack = list()
        stack.append(self.root_node)
        while len(stack) > 0:
            node = stack.pop()
            for child in reversed(node.children):
                stack.append(child)
            if node.depth == query_granularity:
                collected_nodes.append(node)
            if node.depth < query_granularity <= node.granularity:
                collected_nodes.append(node)
        return collected_nodes


class ModelSummaryNode(object):
    """Model summary node."""

    def __init__(self, name: str = '', parent=None):
        self._name = name
        self._input_shape = None
        self._output_shape = None
        self._parameter_quantity = 0
        self._inference_memory = 0
        self._madd = 0
        self._memory = (0, 0)
        self._flops = 0
        self._duration = 0
        self._duration_percent = 0
        self._granularity = 1

        # the amount of depth this node contributes to the tree
        self._depth = 1
        self.parent = parent
        self.children = list()

    @property
    def name(self):
        """Name of summary node."""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def granularity(self):
        """Granularity of the node."""
        return self._granularity

    @granularity.setter
    def granularity(self, granularity):
        self._granularity = granularity

    @property
    def depth(self):
        """Total depth of the node and its children."""
        total_depth = self._depth
        if len(self.children) > 0:
            total_depth += max([child.depth for child in self.children])
        return total_depth

    @property
    def input_shape(self):
        """Input shape of the node or its first child."""
        if len(self.children) == 0:  # leaf
            input_shape = self._input_shape
        else:
            input_shape = self.children[0].input_shape
        return input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        self._input_shape = input_shape

    @property
    def output_shape(self):
        """Output shape of the node or its last child."""
        if len(self.children) == 0:  # leaf
            output_shape = self._output_shape
        else:
            output_shape = self.children[-1].output_shape
        return output_shape

    @output_shape.setter
    def output_shape(self, output_shape):
        assert isinstance(output_shape, (list, tuple))
        self._output_shape = output_shape

    @property
    def parameter_quantity(self):
        """Parameter quantity count of the node and all its children."""
        # return self.parameters_quantity
        total_parameter_quantity = self._parameter_quantity
        for child in self.children:
            total_parameter_quantity += child.parameter_quantity
        return total_parameter_quantity

    @parameter_quantity.setter
    def parameter_quantity(self, parameter_quantity):
        assert parameter_quantity >= 0
        self._parameter_quantity = parameter_quantity

    @property
    def inference_memory(self):
        """Inference memory of the node and all its children."""
        total_inference_memory = self._inference_memory
        for child in self.children:
            total_inference_memory += child.inference_memory
        return total_inference_memory

    @inference_memory.setter
    def inference_memory(self, inference_memory):
        self._inference_memory = inference_memory

    @property
    def madd(self):
        """Total number of multiplication and addition operations."""
        total_madd = self._madd
        for child in self.children:
            total_madd += child.madd
        return total_madd

    @madd.setter
    def madd(self, madd):
        self._madd = madd

    @property
    def flops(self):
        """Total flops."""
        total_flops = self._flops
        for child in self.children:
            total_flops += child.flops
        return total_flops

    @flops.setter
    def flops(self, flops):
        self._flops = flops

    @property
    def memory(self):
        """Total memory usage."""
        total_memory = self._memory
        for child in self.children:
            total_memory[0] += child.memory[0]
            total_memory[1] += child.memory[1]
        return total_memory

    @memory.setter
    def memory(self, memory):
        assert isinstance(memory, (list, tuple))
        self._memory = memory

    @property
    def duration(self):
        """Total duration."""
        total_duration = self._duration
        for child in self.children:
            total_duration += child.duration
        return total_duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    def find_child_index(self, child_name):
        """Find index of a child given its name."""
        assert isinstance(child_name, str)
        index = -1
        for i in range(len(self.children)):
            if child_name == self.children[i].name:
                index = i
        return index

    def add_child(self, node):
        """Add a child node."""
        assert isinstance(node, ModelSummaryNode)
        # make sure no existing child node with the same name
        index = self.find_child_index(node.name)
        if index == -1:  # not exist
            self.children.append(node)
