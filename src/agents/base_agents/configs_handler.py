"""Agent for handling configs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import argparse

from src.configs.config_node import ConfigNode

LOGGER = logging.getLogger(__name__)


class ConfigsHandler:
    """Agent for handling configs."""

    def __init__(self, external_configs_list: list = None):
        # Get configs by overwriting default configs with the order of
        # base_configs, exp_configs, and external_configs
        self.configs = self.get_configs(external_configs_list)

    def get_configs(self, external_configs_list: list = None) -> ConfigNode:
        """Get configs."""
        default_configs = self.get_default_configs()
        command_line_configs = self.get_command_line_configs()
        external_configs = self.get_external_configs(external_configs_list)

        default_configs.merge_from_other_config(command_line_configs)
        default_configs.merge_from_other_config(external_configs)

        return default_configs

    @staticmethod
    def get_command_line_configs() -> ConfigNode:
        """Get command line arguments."""
        args_parser = argparse.ArgumentParser()
        args_parser.add_argument('-c', '--configs', default=None, nargs='*',
                                 help='Optional basic configs of experiments '
                                 'to overwrite the default configs')

        args = args_parser.parse_args()
        command_line_configs = ConfigNode(new_allowed=True)

        if isinstance(args.configs, (tuple, list)):
            for configs_path in args.configs:
                LOGGER.debug('Get configs from: %s', configs_path)
                command_line_configs.merge_from_file(configs_path)

        elif isinstance(args.configs, str):
            LOGGER.debug('Get configs from: %s', args.configs)
            command_line_configs.merge_from_file(args.configs)

        elif args.configs is not None:
            LOGGER.warning('Invalid command line configs: %s', args.configs)

        return command_line_configs

    @staticmethod
    def get_external_configs(external_configs_list=None) -> ConfigNode():
        """Get external configs from a list passed in the init of instance."""
        external_configs = ConfigNode(new_allowed=True)

        if external_configs_list is not None:
            external_configs.merge_from_list(external_configs_list)
            LOGGER.debug('Get external configs: %s', external_configs_list)

        return external_configs

    def save_run_time_configs(self, save_dir: str):
        """Save configs to file in the given directory."""
        configs_path = os.path.join(save_dir, 'run_time_configs.yaml')
        self.configs.dump_to_file(configs_path)
        LOGGER.debug('Saved run time configs to: %s', configs_path)

    @staticmethod
    def get_default_configs() -> ConfigNode:
        """Get default configs."""
        default_configs = ConfigNode(new_allowed=True)
        return default_configs
