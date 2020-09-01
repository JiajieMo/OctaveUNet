"""Agent for handling state of agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import torch

from src.agents.base_agents.metrics_handler import MetricsHandler
from src.agents.base_agents.graph_handler import GraphHandler

LOGGER = logging.getLogger(__name__)


# pylint: disable=abstract-method
class StateHandler(GraphHandler, MetricsHandler):
    """Agent for handling state of agent."""

    def init_agent_state(self):
        """Initialize agent state."""
        # init monitors
        self.reset_monitors()
        # init current epoch and step count
        self.reset_counts()

    def load_agent_state(self, ckpt_path):
        """Load agent states from a given checkpoint path."""
        try:
            ckpt_dict = torch.load(ckpt_path, map_location=self.device)
        except OSError as error:
            ckpt_dict = None
            LOGGER.warning(error)
            LOGGER.warning('Ignoring model checkpoint that failed loading: %s',
                           ckpt_path)

        if ckpt_dict is not None:
            try:
                self.model.load_state_dict(ckpt_dict['model'], strict=True)

            except KeyError as error:
                LOGGER.warning(error)
                LOGGER.warning('Expected key ("model") for state dict of model '
                               '(%s) not found in checkpoint: %s',
                               type(self.model), ckpt_path)
            except RuntimeError as error:
                LOGGER.warning(error)
                LOGGER.warning('State dict of model (%s) do not fully match'
                               'state dict of checkpoint, try non-strictly '
                               'loading checkpoint: %s', type(self.model),
                               ckpt_path)
                self.model.load_state_dict(ckpt_dict['model'], strict=False)

            try:
                self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            except KeyError as error:
                LOGGER.warning(error)
                LOGGER.warning('Expected key ("optimizer") for state dict of '
                               'optimizer (%s) not found in checkpoint: %s',
                               type(self.optimizer), ckpt_path)

            try:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(
                        ckpt_dict['lr_scheduler'])
            except KeyError as error:
                LOGGER.warning(error)
                LOGGER.warning('Expected key ("lr_scheduler") for state dict of '
                               'leanring rate scheduler (%s) not found in '
                               'checkpoint: %s', type(self.lr_scheduler),
                               ckpt_path)

            # always load epoch and step count after successfully load
            self.current_epoch = ckpt_dict['epoch']
            self.current_step = ckpt_dict['step']
            self.monitors = ckpt_dict['monitors']

            LOGGER.info('Successfully loaded agent checkpoint: %s', ckpt_path)
            LOGGER.info('Model checkpoint at training epoch: %d, step: %d',
                        self.current_epoch, self.current_step)

    def save_agent_state(self, ckpt_path):
        """Save agent states into given directory."""
        ckpt_dict = {}

        ckpt_dict['epoch'] = self.current_epoch
        ckpt_dict['step'] = self.current_step
        ckpt_dict['monitors'] = self.monitors
        ckpt_dict['model'] = self.model.state_dict()
        ckpt_dict['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            ckpt_dict['lr_scheduler'] = self.lr_scheduler.state_dict()

        torch.save(ckpt_dict, ckpt_path)

    @staticmethod
    def get_latest_modified_ckpt(ckpt_dir: str, ckpt_ext: str = '.pth') -> str:
        """Get latest modified file in given directory matching extension."""
        ckpt_paths = [os.path.join(ckpt_dir, ckpt_name) for ckpt_name in
                      os.listdir(ckpt_dir) if ckpt_name.endswith(ckpt_ext)]

        if len(ckpt_paths) > 0:
            latest_ckpt_path = max(ckpt_paths, key=os.path.getmtime)
        else:
            latest_ckpt_path = None

        return latest_ckpt_path

    def resume_agent_state(self):
        """Load agent state from existing latest modified checkpoint."""
        ckpt_dir = self.paths['ckpt_dir']
        ckpt_ext = self.configs.LOCAL.CKPT_EXT
        ckpt_path = self.get_latest_modified_ckpt(ckpt_dir, ckpt_ext)

        if ckpt_path is not None:
            LOGGER.info('Resuming from latest modified checkpoint: %s',
                        ckpt_path)
            self.load_agent_state(ckpt_path)
        else:
            LOGGER.info('No checkpoint file ("*%s") found in %s',
                        ckpt_ext, ckpt_dir)

    def save_improved_agent_state(self):
        """Save improved agent state."""
        for monitor_name, monitor_dict in self.monitors.items():
            improved = bool(monitor_dict['epoch'] == self.current_epoch and
                            monitor_dict['step'] == self.current_step)
            if improved:
                ckpt_name = monitor_name + self.configs.LOCAL.CKPT_EXT
                ckpt_path = os.path.join(self.paths['ckpt_dir'], ckpt_name)
                self.save_agent_state(ckpt_path)
