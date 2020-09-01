"""Agent for handling output files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import os
import pandas
import torchvision

from src.agents.base_agents.graph_handler import GraphHandler
from src.agents.base_agents.data_handler import DataHandler


LOGGER = logging.getLogger(__name__)


# pylint: disable=abstract-method
class OutputHandler(GraphHandler, DataHandler):
    """Agent for handling output files."""

    def save_model_summary(self):
        """Save model summary."""
        model_summary_path = os.path.join(self.paths['exp_dir'],
                                          'model_summary.csv')
        if self.model_summary is not None:
            self.model_summary.to_csv(model_summary_path)

    def save_metrics(self, output_dir, sample_ids, loss_meter, metric_meters):
        """Write out files of metrics values."""
        loss_name = self.configs.LOSS.LOSS_NAME
        loss_data_frame = pandas.DataFrame(
            loss_meter.recorded_values, columns=[loss_name])

        metrics_data = {}
        for metric_name, metric_meter in metric_meters.meters.items():
            metrics_data[metric_name] = metric_meter.recorded_values

        metrics_data_frame = pandas.DataFrame(metrics_data)
        sample_ids_data_frame = pandas.DataFrame(
            sample_ids, columns=['sample_index'])

        main_data_frame = pandas.concat((sample_ids_data_frame,
                                         loss_data_frame,
                                         metrics_data_frame),
                                        axis=1, sort=False)

        main_data_frame.to_csv(os.path.join(output_dir, 'performances.csv'))

        main_data_frame.loc['mean'] = main_data_frame.mean()
        main_data_frame.loc['std'] = main_data_frame.std()
        LOGGER.info('metrics:\n%s', main_data_frame.iloc[
            -2:, main_data_frame.columns != 'sample_index'].to_string())

    def save_figures(self, output_dir: str, sample_id: int, **figures):
        """Write out figures."""
        for figure_name, figure in figures.items():
            if figure is not None:
                torchvision.utils.save_image(
                    tensor=figure,
                    filename=os.path.join(output_dir, '{}-{}.png'.format(
                        sample_id, figure_name))
                )
