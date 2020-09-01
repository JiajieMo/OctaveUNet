"""Model summary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from copy import deepcopy
from collections import OrderedDict
from torch import nn

from src.models.utils.model_summary.model_summary_tree import ModelSummaryNode
from src.models.utils.model_summary.model_summary_tree import ModelSummaryTree
from src.models.utils.model_summary.model_summary_formatter import format_model_summary
from src.models.utils.model_summary.model_summary_hook import ModelSummaryHook

LOGGER = logging.getLogger(__name__)


def get_parent_node(root_node, node_name):
    """Get parent node of the given node name."""
    assert isinstance(root_node, ModelSummaryNode)
    node = root_node
    names = node_name.split('.')
    for i in range(len(names) - 1):
        node_name = '.'.join(names[0:i+1])
        child_index = node.find_child_index(node_name)
        assert child_index != -1
        node = node.children[child_index]
    return node


def convert_leaf_modules_to_summary_tree(leaf_modules):
    """Convert leaf modules to tree."""
    assert isinstance(leaf_modules, OrderedDict)

    create_index = 1
    root_node = ModelSummaryNode(name='root', parent=None)
    for leaf_module_name, leaf_module in leaf_modules.items():
        names = leaf_module_name.split('.')
        for i in range(len(names)):
            create_index += 1
            summary_node_name = '.'.join(names[0:i+1])
            parent_node = get_parent_node(root_node, summary_node_name)
            node = ModelSummaryNode(name=summary_node_name, parent=parent_node)
            parent_node.add_child(node)
            if i == len(names) - 1:  # leaf module itself
                input_shape = leaf_module.input_shape.numpy().tolist()
                output_shape = leaf_module.output_shape.numpy().tolist()
                node.input_shape = input_shape
                node.output_shape = output_shape
                node.parameter_quantity = \
                    leaf_module.parameter_quantity.numpy()[0]
                node.inference_memory = leaf_module.inference_memory.numpy()[0]
                node.madd = leaf_module.madd.numpy()[0]
                node.flops = leaf_module.flops.numpy()[0]
                node.duration = leaf_module.duration.numpy()[0]
                node.memory = leaf_module.memory.numpy().tolist()

    return ModelSummaryTree(root_node)


class ModelSummary():
    """Model summary."""

    def __init__(self, model, input_size, query_granularity=1):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (tuple, list)) and len(input_size) == 3
        self._model = deepcopy(model)
        self._input_size = input_size
        self._query_granularity = query_granularity

    def _analyze_model(self):
        model_hook = ModelSummaryHook(self._model, self._input_size)
        leaf_modules = model_hook.retrieve_leaf_modules()
        # do not forget to unhook the model
        model_hook.unhook_model()
        summary_tree = convert_leaf_modules_to_summary_tree(leaf_modules)
        collected_nodes = summary_tree.get_collected_summary_nodes(
            self._query_granularity)
        return collected_nodes

    def get_data_frame(self):
        """Show model summary."""
        collected_nodes = self._analyze_model()
        summary_df = format_model_summary(collected_nodes)
        return summary_df
