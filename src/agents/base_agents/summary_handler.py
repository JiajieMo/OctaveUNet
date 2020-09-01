"""Agent for handling tensorboard summary writer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import time
import os
import torch

from torch.utils.tensorboard import SummaryWriter

from src.agents.base_agents.graph_handler import GraphHandler
from src.agents.base_agents.metrics_handler import MetricsHandler


LOGGER = logging.getLogger(__name__)


# pylint: disable=abstract-method
class SummaryHandler(GraphHandler, MetricsHandler):
    """Agent for handling tensorboard summary writer."""

    def __init__(self, external_configs_list: list = None):
        # init configs
        super(SummaryHandler, self).__init__(external_configs_list)

        # summary writer
        self.summ_writer = None

    def get_summ_writer(self) -> torch.utils.tensorboard.SummaryWriter:
        """Initialize summary writer for using tensorboard."""
        summ_writer = SummaryWriter(
            log_dir=self.paths['summ_dir'], purge_step=self.current_epoch)
        return summ_writer

    def write_learning_rate(self):
        """Write learning rate."""
        self.summ_writer.add_scalar('learning_rate',
                                    self.optimizer.param_groups[0]['lr'],
                                    self.current_epoch,
                                    time.time())

    def write_metrics(self, loss_meter, metric_meters, prefix: str = 'train'):
        """Write both training and validating (if provided) loss and metrics."""
        self.summ_writer.add_scalar(os.path.join(prefix, 'loss'),
                                    loss_meter.average_value,
                                    self.current_epoch,
                                    time.time())

        for metric in metric_meters.names:
            self.summ_writer.add_scalar(
                tag=os.path.join(prefix, metric),
                scalar_value=metric_meters.average_values[metric],
                global_step=self.current_epoch,
                walltime=time.time()
            )

    def write_figures(self, prefix: str, **figures):
        """Write figures."""
        for figure_name, figure in figures.items():
            if figure is not None:
                self.summ_writer.add_images(
                    tag=os.path.join(prefix, figure_name),
                    img_tensor=figure,
                    global_step=self.current_epoch,
                    walltime=time.time(),
                    dataformats='NCHW',
                )

    def write_comparisons(self, tarin_loss_meter, train_metric_meters,
                          valid_loss_meter=None, valid_metric_meters=None):
        """Write loss and metrics comparison of training and validating."""
        if valid_loss_meter is not None:
            self.summ_writer.add_scalars(
                main_tag='loss',
                tag_scalar_dict={'train': tarin_loss_meter.average_value,
                                 'valid': valid_loss_meter.average_value},
                global_step=self.current_epoch,
                walltime=time.time(),
            )

        else:
            self.summ_writer.add_scalars(
                main_tag='loss',
                tag_scalar_dict={'train': tarin_loss_meter.average_value},
                global_step=self.current_epoch,
                walltime=time.time(),
            )

        if valid_metric_meters is not None:
            for train_metric, valid_metric in zip(train_metric_meters.names,
                                                  valid_metric_meters.names):
                assert train_metric == valid_metric
                self.summ_writer.add_scalars(
                    main_tag=train_metric,
                    tag_scalar_dict={
                        'train': train_metric_meters.average_values[
                            train_metric],
                        'valid': valid_metric_meters.average_values[
                            valid_metric],
                    },
                    global_step=self.current_epoch,
                    walltime=time.time(),
                )

        else:
            for train_metric in train_metric_meters.names:
                self.summ_writer.add_scalars(
                    main_tag=train_metric,
                    tag_scalar_dict={
                        'train': train_metric_meters.average_values[
                            train_metric]
                    },
                    global_step=self.current_epoch,
                    walltime=time.time(),
                )
