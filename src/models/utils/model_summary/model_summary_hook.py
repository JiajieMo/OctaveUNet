"""Module hook for model summary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from src.models.utils.model_summary.utils.compute_flops import compute_flops
from src.models.utils.model_summary.utils.compute_madd import compute_madd
from src.models.utils.model_summary.utils.compute_memory import compute_memory


LOGGER = logging.getLogger(__name__)


class ModelSummaryHook():
    """Model summary hook."""

    def __init__(self, model, input_size):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (list, tuple))

        self._model = model
        self._input_size = input_size
        self._origin_call = dict()  # sub module call hook
        self._hooked_call = dict()

        self._hook_model()
        inputs = torch.rand(1, *self._input_size)  # add module duration time
        self._model.eval()
        self._model(inputs)

    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)

        if len(list(module.children())) > 0:
            pass
        else:
            module.register_buffer('input_shape', torch.zeros(3).int())
            module.register_buffer('output_shape', torch.zeros(3).int())
            module.register_buffer('parameter_quantity', torch.zeros(1).int())
            module.register_buffer('inference_memory', torch.zeros(1).long())
            module.register_buffer('madd', torch.zeros(1).long())
            module.register_buffer('duration', torch.zeros(1).float())
            module.register_buffer('flops', torch.zeros(1).long())
            module.register_buffer('memory', torch.zeros(2).long())

    def _sub_module_call_hook(self):
        def wrap_call(module, *inputs, **kwargs):
            assert module.__class__ in self._origin_call

            # item size for memory
            itemsize = inputs[0].detach().numpy().itemsize

            start = time.time()
            output = self._origin_call[module.__class__](
                module, *inputs, **kwargs)
            end = time.time()
            module.duration = torch.from_numpy(
                np.array([end - start], dtype=np.float32))

            module.input_shape = torch.from_numpy(
                np.array(inputs[0].size()[1:], dtype=np.int32))
            module.output_shape = torch.from_numpy(
                np.array(output.size()[1:], dtype=np.int32))

            parameter_quantity = 0
            # iterate through parameters and count num params
            for param in module.parameters():
                parameter_quantity += (0 if param is None else torch.numel(
                    param.data))

            module.parameter_quantity = torch.from_numpy(
                np.array([parameter_quantity], dtype=np.long))

            inference_memory = 1
            for i in output.size()[1:]:
                inference_memory *= i
            # memory += parameters_number  # exclude parameter memory

            # memory in MB unit
            inference_memory = inference_memory * 4 / (1024 ** 2)
            module.inference_memory = torch.from_numpy(
                np.array([inference_memory], dtype=np.float32))

            if len(inputs) == 1:
                madd = compute_madd(module, inputs[0], output)
                flops = compute_flops(module, inputs[0], output)
                memory = compute_memory(module, inputs[0], output)

            elif len(inputs) > 1:
                madd = compute_madd(module, inputs, output)
                flops = compute_flops(module, inputs, output)
                memory = compute_memory(module, inputs, output)

            else:  # error
                LOGGER.debug('Assuming module of type: %s with unknown number '
                             'of inputs: %d to have 0 MAdd, 0 FLOPs and 0 '
                             'reading and writing memory',
                             type(module), len(inputs))
                madd = 0
                flops = 0
                memory = (0, 0)

            module.madd = torch.from_numpy(np.array([madd], dtype=np.int64))
            module.flops = torch.from_numpy(np.array([flops], dtype=np.int64))
            memory = np.array(memory, dtype=np.int32) * itemsize
            module.memory = torch.from_numpy(memory)

            return output

        for module in self._model.modules():
            if (len(list(module.children())) == 0 and
                    module.__class__ not in self._origin_call):
                self._origin_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = wrap_call

    def _sub_module_call_unhook(self):
        for module in self._model.modules():
            if (len(list(module.children())) == 0 and
                    module.__class__ not in self._hooked_call):
                self._hooked_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = self._origin_call[module.__class__]

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        self._sub_module_call_hook()

    def unhook_model(self):
        """Unhook model when nodes are collected."""
        self._sub_module_call_unhook()

    @staticmethod
    def _retrieve_leaf_modules(model):
        leaf_modules = list()
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                leaf_modules.append((name, module))
        return leaf_modules

    def retrieve_leaf_modules(self):
        """Retrieve leaf modules."""
        return OrderedDict(self._retrieve_leaf_modules(self._model))
