"""Agent for handling computation graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

from src.models.utils.model_summary.model_summary import ModelSummary

from src.agents.base_agents.configs_handler import ConfigsHandler
from src.agents.base_agents.paths_handler import PathsHandler
from src.agents.base_agents.device_handler import DeviceHandler
from src.agents.base_agents.data_handler import DataHandler

LOGGER = logging.getLogger(__name__)


# pylint: disable=abstract-method
class GraphHandler(DataHandler, DeviceHandler, PathsHandler, ConfigsHandler):
    """Agent for handling computation graph."""

    def __init__(self, external_configs_list: list = None):
        # init configs
        super(GraphHandler, self).__init__(external_configs_list)

        # computation graph related objects
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None
        self.model_summary = None

    def handle_computation_graph(self):
        """Get model, criterion, optimizer, and lr_scheduler."""
        # Get model
        self.model = self.get_model()

        # Save model summary
        self.model_summary = self.get_model_summary()

        # Get loss criterion
        self.criterion = self.get_criterion()

        # Deploy computation graph to devices, and handling data parallel
        self.deploy_to_device()

        # Get optimizer after deploied to devices
        self.optimizer = self.get_optimizer()

        # Get lr_scheduler after optimizer is initialized
        self.lr_scheduler = self.get_lr_scheduler()

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer."""
        optimizer_name = self.configs.OPTIM.OPTIMIZER_NAME
        learning_rate = self.configs.OPTIM.LEARNING_RATE

        if optimizer_name == 'sgd':
            momentum = self.configs.OPTIM.MOMENTUM
            if momentum is None:
                momentum = 0

            weight_decay = self.configs.OPTIM.WEIGHT_DECAY
            if weight_decay is None:
                weight_decay = 0

            optimizer = torch.optim.SGD(params=self.model.parameters(),
                                        lr=learning_rate,
                                        momentum=momentum,
                                        dampening=0,
                                        weight_decay=weight_decay,
                                        nesterov=False)

        elif optimizer_name == 'adam':
            weight_decay = self.configs.OPTIM.WEIGHT_DECAY
            if weight_decay is None:
                weight_decay = 0

            optimizer = torch.optim.Adam(params=self.model.parameters(),
                                         lr=learning_rate,
                                         betas=(0.9, 0.999),
                                         eps=1e-8,
                                         weight_decay=weight_decay,
                                         amsgrad=False)

        else:
            LOGGER.error('Invalid optimizer: %s', optimizer_name)
            raise NotImplementedError

        LOGGER.info('Retrieved optimizer: %s', optimizer_name)

        return optimizer

    def get_lr_scheduler(self):
        """Get learning rate scheduler."""
        try:
            scheduler_name = self.configs.OPTIM.LR_SCHEDULER.SCHEDULER_NAME
        except AttributeError:
            scheduler_name = None

        if scheduler_name is None:
            lr_scheduler = None

        elif scheduler_name == 'plateau':
            factor = self.configs.OPTIM.LR_SCHEDULER.FACTOR
            patience = self.configs.OPTIM.LR_SCHEDULER.PATIENCE

            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                verbose=False,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=0,
                min_lr=1e-6,  # default min_lr is 0
                eps=1e-8,
            )

        elif scheduler_name == 'cyclic':
            base_lr = self.configs.OPTIM.LR_SCHEDULER.BASE_LR
            max_lr = self.configs.OPTIM.LR_SCHEDULER.MAX_LR

            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer=self.optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=2000,
                step_size_down=None,
                mode='triangular',
                gamma=1.0,
                scale_fn=None,
                scale_mode='cycle',
                cycle_momentum=True,
                base_momentum=0.8,
                max_momentum=0.9,
                last_epoch=-1,
            )

        else:
            LOGGER.error('Invalid learning rate scheduler: %s', scheduler_name)
            raise NotImplementedError

        return lr_scheduler

    def deploy_to_device(self):
        """Deploy computation graph on devices."""
        if self.device_ids is not None and len(self.device_ids) > 1:
            if not isinstance(self.model, torch.nn.DataParallel):
                self.model = torch.nn.DataParallel(self.model, self.device_ids)

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def get_model_summary(self, query_granularity: int = 1):
        """Record model summary."""
        data_shape = self.get_data_shape(self.train_loader, self.image_key)[1:]
        model_summary = ModelSummary(self.model, data_shape, query_granularity)
        model_summary = model_summary.get_data_frame()
        return model_summary

    def get_model(self) -> torch.nn.Module:
        """Get model."""
        raise NotImplementedError

    def get_criterion(self) -> torch.nn.Module:
        """Get loss criterion."""
        raise NotImplementedError
