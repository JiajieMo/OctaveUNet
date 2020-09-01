"""Agent for handling device."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

from src.agents.base_agents.configs_handler import ConfigsHandler

LOGGER = logging.getLogger(__name__)


class DeviceHandler(ConfigsHandler):
    """Agent for handling device."""

    def __init__(self, external_configs_list: list = None):
        # init configs
        super(DeviceHandler, self).__init__(external_configs_list)

        # device and device_ids
        self.device = None
        self.device_ids = None

    def handle_device(self):
        """Get device, setup random seed and benchmark."""
        # get device and device_ids from configs
        self.device, self.device_ids = self.get_device()

        # set random seed from configs
        self.set_random_seed()

        # set device benchmark accroding to configs,
        # should only use when input shapes are consistent
        self.set_device_benchmark()

    def get_device(self):
        """Get device for agent."""
        enable_cuda = bool(self.configs.DEVICE.ENABLE_CUDA and
                           torch.cuda.is_available())

        if enable_cuda:
            devices_ids = self.configs.DEVICE.DEVICE_IDS
            if isinstance(devices_ids, int):
                device = torch.device('cuda:{}'.format(devices_ids))
                devices_ids = [devices_ids, ]

            elif isinstance(devices_ids, (list, tuple)):
                assert all(isinstance(idx, int) for idx in devices_ids)
                assert torch.cuda.device_count() >= len(devices_ids)
                device = torch.device('cuda:{}'.format(devices_ids[0]))
                devices_ids = list(devices_ids)

            else:
                LOGGER.error('Invalid device ids: %s', devices_ids)
                raise TypeError(devices_ids)

            LOGGER.info('Agent will run on device(s): %s', devices_ids)

        else:
            LOGGER.info('Agent will run on CPU only')
            device = torch.device('cpu')
            devices_ids = None

        return device, devices_ids

    def set_random_seed(self):
        """Set random seed for both CPU or GPU(optional)."""
        try:
            random_seed = self.configs.DEVICE.RANDOM_SEED
        except AttributeError:
            random_seed = None

        if isinstance(random_seed, int):
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            LOGGER.info('Set random seed: %d', random_seed)

        else:
            LOGGER.debug('Ignoring invalid random seed: %s', random_seed)

    def set_device_benchmark(self):
        """Run optimization when inputs are with the same shape."""
        if self.configs.DEVICE.ENABLE_BENCHMARK is True:
            torch.backends.cudnn.benchmark = True
