"""Agent for handling metrics and monitors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from src.metrics.value_meters import AverageMeter
from src.metrics.value_meters import AverageMeters

from src.agents.base_agents.configs_handler import ConfigsHandler
from src.agents.base_agents.paths_handler import PathsHandler

LOGGER = logging.getLogger(__name__)


class MetricsHandler(PathsHandler, ConfigsHandler):
    """Agent for handling metrics and monitors."""

    def __init__(self, external_configs_list: list = None):
        # init configs
        super(MetricsHandler, self).__init__(external_configs_list)

        # init metrics and monitors related objects
        self.monitors = None
        self.current_epoch = None
        self.current_step = None

    def reset_counts(self):
        """Reset epoch and step counts."""
        self.current_epoch = 0
        self.current_step = 0

    def reset_monitors(self):
        """Initialize monitors for metrics."""
        monitor_names = self.configs.METRICS.MONITOR_NAMES
        metric_names = self.configs.METRICS.METRIC_NAMES
        self.monitors = {}

        try:
            for monitor_name in monitor_names:
                if monitor_name in metric_names:
                    self.monitors[monitor_name] = {'value': 0.0,
                                                   'epoch': 0,
                                                   'step': 0}
                elif 'loss' in monitor_name:
                    self.monitors[monitor_name] = {'value': float('inf'),
                                                   'epoch': 0,
                                                   'step': 0}
                else:
                    LOGGER.error('Invalid monitor: %s', monitor_name)
                    raise NotImplementedError(monitor_name)
        except:
            pass

    def update_monitors(self, loss_meter: AverageMeter,
                        metric_meters: AverageMeters):
        """Update monitors of metrics and loss criterion."""
        for monitor_name, monitor_dict in self.monitors.items():
            try:
                if 'loss' in monitor_name:
                    current_value = loss_meter.average_value
                    improved = monitor_dict['value'] > current_value

                else:
                    current_value = metric_meters.average_values[monitor_name]
                    improved = monitor_dict['value'] < current_value

            except KeyError:
                LOGGER.error('Invalid monitor name: %s', monitor_name)
                raise ValueError

            if improved:
                # update monitor value
                monitor_dict['value'] = current_value
                monitor_dict['epoch'] = self.current_epoch
                monitor_dict['step'] = self.current_step

    def get_init_meters(self):
        """Get initialized loss and metrics meters."""
        loss_meter = AverageMeter(self.configs.LOSS.LOSS_NAME)
        metric_meters = AverageMeters(self.configs.METRICS.METRIC_NAMES)
        return loss_meter, metric_meters

    def get_metrics(self) -> AverageMeters:
        """Get metric values."""
        raise NotImplementedError
