"""Compute memory cost of each leaf module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from torch import nn

LOGGER = logging.getLogger(__name__)


def compute_memory(module, inp, out):
    """Compute memory usage of the given module."""
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU)):
        module_memory = compute_ReLU_memory(module, inp, out)

    elif isinstance(module, nn.PReLU):
        module_memory = compute_PReLU_memory(module, inp, out)

    elif isinstance(module, nn.Conv2d):
        module_memory = compute_Conv2d_memory(module, inp, out)

    elif isinstance(module, nn.ConvTranspose2d):
        module_memory = compute_ConvTranspose2d_memory(module, inp, out)

    elif isinstance(module, nn.BatchNorm2d):
        module_memory = compute_BatchNorm2d_memory(module, inp, out)

    elif isinstance(module, nn.Linear):
        module_memory = compute_Linear_memory(module, inp, out)

    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        module_memory = compute_Pool2d_memory(module, inp, out)

    else:
        module_memory = (0, 0)

    return module_memory


def get_num_train_params(module):
    """Get number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# pylint: disable=invalid-name
def compute_ReLU_memory(module, inp, out):
    """Compute memory usage of ReLU."""
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU))
    mread = inp.numel()
    mwrite = out.numel()
    return mread, mwrite


def compute_PReLU_memory(module, inp, out):
    """Compute memory usage of PReLU."""
    assert isinstance(module, nn.PReLU)
    batch_size = inp.size()[0]
    mread = batch_size * (inp[0].numel() + get_num_train_params(module))
    mwrite = out.numel()
    return mread, mwrite


def compute_ConvTranspose2d_memory(module, inp, out):
    """Compute memory usage of ConvTranspose2d."""
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.ConvTranspose2d)
    assert len(inp.size()) == 4
    assert len(out.size()) == 4

    batch_size = inp.size()[0]
    # This includes weights with bias if the module contains it.
    mread = batch_size * (inp[0].numel() + get_num_train_params(module))
    mwrite = out.numel()
    return mread, mwrite


def compute_Conv2d_memory(module, inp, out):
    """Compute memory usage of Conv2d."""
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4
    assert len(out.size()) == 4

    batch_size = inp.size()[0]
    # This includes weights with bias if the module contains it.
    mread = batch_size * (inp[0].numel() + get_num_train_params(module))
    mwrite = out.numel()
    return mread, mwrite


def compute_BatchNorm2d_memory(module, inp, out):
    """Compute memory usage of BatchNorm2d."""
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4
    assert len(out.size()) == 4
    batch_size, in_c, *_ = inp.size()

    mread = batch_size * (inp[0].numel() + 2 * in_c)
    mwrite = out.numel()
    return mread, mwrite


def compute_Linear_memory(module, inp, out):
    """Compute memory usage of Linear."""
    assert isinstance(module, nn.Linear)
    assert len(inp.size()) == 2
    assert len(out.size()) == 2
    batch_size = inp.size()[0]
    # This includes weights with bias if the module contains it.
    mread = batch_size * (inp[0].numel() + get_num_train_params(module))
    mwrite = out.numel()

    return mread, mwrite


def compute_Pool2d_memory(module, inp, out):
    """Compute memory usage of MaxPool2d or AvgPool2d."""
    assert isinstance(module, (nn.MaxPool2d, nn.AvgPool2d))
    assert len(inp.size()) == 4
    assert len(out.size()) == 4
    mread = inp.numel()
    mwrite = out.numel()
    return mread, mwrite
