"""Agent for handling paths."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import datetime
import os

from src.agents.base_agents.configs_handler import ConfigsHandler

LOGGER = logging.getLogger(__name__)


class PathsHandler(ConfigsHandler):
    """Agent for handling commently sued paths."""

    def __init__(self, external_configs_list: list = None):
        # init configs
        super(PathsHandler, self).__init__(external_configs_list)
        # init PathsHandler
        self.paths = None

    def handle_paths(self):
        """Setup paths, make sure paths exist and backup configs."""
        # setup paths with exp_dir, log_dir, summ_dir, out_dir, and ckpt_dir
        self.get_paths()

        # create dirs if not exist
        self.make_sure_paths_exist()

        # save run time configs
        self.save_run_time_configs(self.paths['exp_dir'])

        # make configuration immutable
        self.configs.freeze()

    def get_paths(self):
        """Set paths from configs."""
        self.paths = {}

        exp_subdir = self.configs.LOCAL.EXP_SUBDIR
        if exp_subdir is None:
            exp_subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.configs.LOCAL.EXP_SUBDIR = exp_subdir

        self.paths['exp_dir'] = os.path.join(self.configs.LOCAL.EXP_ROOT,
                                             exp_subdir)
        self.paths['log_dir'] = os.path.join(self.paths['exp_dir'],
                                             self.configs.LOCAL.LOG_SUBDIR)
        self.paths['summ_dir'] = os.path.join(self.paths['exp_dir'],
                                              self.configs.LOCAL.SUMM_SUBDIR)
        self.paths['out_dir'] = os.path.join(self.paths['exp_dir'],
                                             self.configs.LOCAL.OUT_SUBDIR)
        self.paths['ckpt_dir'] = os.path.join(self.paths['exp_dir'],
                                              self.configs.LOCAL.CKPT_SUBDIR)

    def make_sure_paths_exist(self):
        """Make sure paths exist."""
        for path_name, path in self.paths.items():
            if self.make_sure_path_exist(path):
                LOGGER.debug('Created directory of %s in: %s', path_name, path)
            else:
                LOGGER.debug('Directory of %s already exist: %s',
                             path_name, path)

    @staticmethod
    def make_sure_path_exist(path):
        """Recursively create path if not already exist."""
        if not os.path.isdir(path):
            os.makedirs(path)
            return True
        return False
