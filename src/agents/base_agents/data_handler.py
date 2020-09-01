"""Agent for handling data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

from src.agents.base_agents.configs_handler import ConfigsHandler
from src.agents.base_agents.paths_handler import PathsHandler

LOGGER = logging.getLogger(__name__)


class DataHandler(PathsHandler, ConfigsHandler):
    """Agent for handling data."""

    def __init__(self, external_configs_list: list = None):
        # init configs
        super(DataHandler, self).__init__(external_configs_list)

        # init sample keys
        self.image_key = None
        self.target_key = None
        self.mask_key = None

        # data loaders
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def handle_data(self):
        """Get data loaders and sample keys."""
        self.image_key, self.target_key, self.mask_key = self.get_sample_keys()
        (self.train_loader, self.valid_loader,
         self.test_loader) = self.get_data_loaders()

    def get_sample_keys(self) -> (str, str, str):
        """Get sample keys for reading data of image, target, and mask."""
        # get keys of data sample
        image_key = self.configs.DATA.DATASET.IMAGE_KEY
        target_key = self.configs.DATA.DATASET.TARGET_KEY

        # not all datasets have mask for region of interest
        try:
            mask_key = self.configs.DATA.DATASET.MASK_KEY
        except AttributeError:
            mask_key = None

        return image_key, target_key, mask_key

    @staticmethod
    def get_data_shape(data_loader: torch.utils.data.DataLoader,
                       data_key: str) -> [int, int, int, int]:
        """Get data shape from data loader."""
        data_shape = next(iter(data_loader))[data_key].shape
        return data_shape

    def get_data_loaders(self) -> (torch.utils.data.DataLoader,
                                   torch.utils.data.DataLoader,
                                   torch.utils.data.DataLoader):
        """Get custom train, valid, and test data loader."""
        raise NotImplementedError
