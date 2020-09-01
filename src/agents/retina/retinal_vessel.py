"""Agent for retinal vessel segmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import tqdm

import numpy
import torch

from src.agents.base_agents.output_handler import OutputHandler
from src.agents.base_agents.state_handler import StateHandler
from src.agents.base_agents.summary_handler import SummaryHandler

from src.agents.retina.utils.get_data import get_data_loader
from src.agents.retina.utils.get_metrics import get_metrics
from src.agents.retina.utils.get_criterion import get_criterion
from src.agents.retina.utils.get_model import get_model

from src.processings.thresholdings import batch_thresholding

from src.metrics import binary_confusion


LOGGER = logging.getLogger(__name__)


# pylint: disable=too-many-ancestors
class RetinalVesselSegmentation(SummaryHandler, StateHandler, OutputHandler):
    """Retinal vessel segmentation."""

    def __init__(self, external_configs_list: list = None):
        super(RetinalVesselSegmentation, self).__init__(external_configs_list)

        # setup paths, make sure paths exist and backup configs
        self.handle_paths()

        # get sample keys and data loaders
        self.handle_data()

        # get device and setup random seed and benchmark
        self.handle_device()

        # get model, criterion, optimizer, and lr_scheduler
        self.handle_computation_graph()

        # save model summary
        self.save_model_summary()

        # initialize agent state of epoch count, step count, and monitors
        self.init_agent_state()

        if self.configs.AGENT.RESUME is True:
            # resume from latest modified ckpt
            self.resume_agent_state()

        # get summary writer after resumed
        self.summ_writer = self.get_summ_writer()

        # write computation graph
        self.summ_writer.add_graph(
            self.model, next(iter(self.train_loader))[self.image_key].to(
                self.device))

    def run(self):
        """Running routine of agent."""
        run_mode = self.configs.AGENT.RUN_MODE
        try:
            if run_mode == 'train':
                self.run_training()

            elif run_mode == 'test':
                self.run_testing()

            else:
                LOGGER.error('Invalid agent run mode: %s', run_mode)
                raise NotImplementedError

        except KeyboardInterrupt:
            LOGGER.warning('Agent interrupted by KeyboardInterrupt')

        self.run_finalizing()

    @staticmethod
    def get_progress_bar(iterable, desc, initial=0, total=None,
                         remain_on_screen=False):
        """Get progress bar."""
        if isinstance(iterable, torch.utils.data.DataLoader):
            dataset_size = len(iterable.dataset)
            batch_size = iterable.batch_size
            drop_last = iterable.drop_last
            num_steps = dataset_size / batch_size
            num_steps = int(num_steps) if drop_last else int(
                numpy.ceil(num_steps))

            initial = 0
            total = num_steps

        progress_bar = tqdm.tqdm(iterable=iterable, desc=desc,
                                 initial=initial, total=total,
                                 leave=remain_on_screen)
        return progress_bar

    def run_training(self):
        """Run training process of agent."""
        train_progress = self.get_progress_bar(
            iterable=range(self.current_epoch, self.configs.OPTIM.MAX_EPOCH),
            desc='Training progress',
            initial=self.current_epoch,
            total=self.configs.OPTIM.MAX_EPOCH,
            remain_on_screen=True,
        )

        for _ in train_progress:
            # write learning rate of current epoch before scheduler is called
            self.write_learning_rate()

            # take one step of training
            train_loss_meter, train_metric_meters = self.step_training()
            # write loss and metrics of training
            self.write_metrics(train_loss_meter, train_metric_meters, 'train')

            if bool(self.current_epoch % self.configs.METRICS.VALID_PATIENCE ==
                    (self.configs.METRICS.VALID_PATIENCE - 1)):
                # evaluate the model once on the validation set
                valid_loss_meter, valid_metric_meters = self.run_validating()

                # write loss and metrics of training
                self.write_metrics(
                    valid_loss_meter, valid_metric_meters, 'valid')

                # update monitored metrics of validation
                self.update_monitors(valid_loss_meter, valid_metric_meters)

                # save agent state if the updated metrics improved
                self.save_improved_agent_state()

                # loss value averaged over an epoch to check if plateaued
                if isinstance(self.lr_scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(valid_loss_meter.average_value)

            else:
                valid_loss_meter = None
                valid_metric_meters = None

            # write comparison curves of training and validating
            compare_patience = self.configs.SUMM.COMPARE_PATIENCE
            if compare_patience is not None and (
                    self.current_epoch % compare_patience) == (
                        compare_patience - 1):
                self.write_comparisons(train_loss_meter, train_metric_meters,
                                       valid_loss_meter, valid_metric_meters)

            # for other lr_scheduler types, step once each epoch
            if self.lr_scheduler is not None and not isinstance(
                    self.lr_scheduler, (
                        torch.optim.lr_scheduler.CyclicLR,
                        torch.optim.lr_scheduler.ReduceLROnPlateau
                    )
            ):
                self.lr_scheduler.step()

            self.current_epoch += 1

    def step_training(self):
        """Run one step of training."""
        # init meters for loss value and metric values
        loss_meter, metric_meters = self.get_init_meters()

        # whether or not to record visual summary
        enable_write_figures = bool(
            self.current_epoch % self.configs.SUMM.FIGURE.TRAIN_PATIENCE == (
                self.configs.SUMM.FIGURE.TRAIN_PATIENCE - 1))

        # get data loader for training
        epoch_progress = self.get_progress_bar(
            iterable=self.train_loader,
            desc='Training epoch {}'.format(self.current_epoch),
            remain_on_screen=False,
        )

        # turn on training mode for model
        self.model.train()

        for sample_batch in epoch_progress:
            images = sample_batch[self.image_key].to(self.device)
            targets = sample_batch[self.target_key].to(self.device)
            masks = None if self.mask_key is None else sample_batch[
                self.mask_key].to(self.device)

            # model outputs inactivated logits
            logits = self.model(images)
            prob_maps = torch.sigmoid(logits)
            binary_maps = self.get_binary_maps(prob_maps)
            analytic_maps = self.get_analytic_maps(binary_maps, targets, masks)

            # criterion inputs inactivated logits
            current_loss = self.criterion(logits, targets)

            if torch.isnan(current_loss):
                LOGGER.warning('Criterion (%s) reached NaN during validation '
                               'at training epoch %d, training step %d',
                               self.configs.LOSS.LOSS_NAME,
                               self.current_epoch, self.current_step)
                continue

            # update model
            self.optimizer.zero_grad()
            current_loss.backward()
            self.optimizer.step()

            # update meters
            loss_meter.accumulate(current_loss.item())
            current_metrics = self.get_metrics(
                prob_maps, binary_maps, targets, masks)
            metric_meters.accumulate(current_metrics)

            # record visual summary
            if enable_write_figures:
                self.write_figures('train', **{
                    'images': images,
                    'targets': targets,
                    'masks': masks,
                    'probability_maps': prob_maps,
                    'binary_maps': binary_maps,
                    'analytic_maps': analytic_maps,
                })

            # cyclic learning rate steps every batch instead of epoch
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
                self.lr_scheduler.step()

            self.current_step += 1

        epoch_progress.close()

        return loss_meter, metric_meters

    @torch.no_grad()
    def run_validating(self):
        """Run validating process of agent."""
        # init meters for loss value and metric values
        loss_meter, metric_meters = self.get_init_meters()

        # whether or not to record visual summary
        enable_write_figures = bool(
            (self.configs.AGENT.RUN_MODE == 'valid') or
            (self.current_epoch % self.configs.SUMM.FIGURE.VALID_PATIENCE == (
                self.configs.SUMM.FIGURE.VALID_PATIENCE - 1))
        )

        # get data loader for validating
        epoch_progress = self.get_progress_bar(
            iterable=self.valid_loader,
            desc='Validating epoch {}'.format(self.current_epoch),
            remain_on_screen=False,
        )

        # turn on evaluation mode for model
        self.model.eval()
        # epoch loop over dataset once
        for sample_batch in epoch_progress:
            images = sample_batch[self.image_key].to(self.device)
            targets = sample_batch[self.target_key].to(self.device)
            masks = None if self.mask_key is None else sample_batch[
                self.mask_key].to(self.device)

            # model outputs unactivated logits
            logits = self.model(images)
            prob_maps = torch.sigmoid(logits)
            binary_maps = self.get_binary_maps(prob_maps)
            analytic_maps = self.get_analytic_maps(binary_maps, targets, masks)

            # criterion inputs unactivated logits
            current_loss = self.criterion(logits, targets)

            if torch.isnan(current_loss):
                LOGGER.warning('Criterion (%s) reached NaN during validation '
                               'at training epoch %d, training step %d',
                               self.configs.LOSS.LOSS_NAME,
                               self.current_epoch, self.current_step)
                continue

            # update meters
            loss_meter.accumulate(current_loss.item())
            current_metrics = self.get_metrics(
                prob_maps, binary_maps, targets, masks)
            metric_meters.accumulate(current_metrics)

            # record visual summary
            if enable_write_figures:
                self.write_figures('valid', **{
                    'images': images,
                    'targets': targets,
                    'masks': masks,
                    'probability_maps': prob_maps,
                    'binary_maps': binary_maps,
                    'analytic_maps': analytic_maps,
                })

        # close tqdm wrapped data loader
        epoch_progress.close()

        return loss_meter, metric_meters

    # pylint: disable=too-many-locals
    @torch.no_grad()
    def run_testing(self, ckpt_path=None, output_dir=None):
        """Run testing process of agent."""

        # load checkpoint for testing or anounce the current checkpoint
        if ckpt_path is not None:
            self.load_agent_state(ckpt_path)
        else:
            LOGGER.info('Model checkpoint at training epoch %d, step %d',
                        self.current_epoch, self.current_step)

        # update outputs directory
        if output_dir is None:
            output_dir = os.path.join(
                self.paths['out_dir'], 'epoch_{}-step_{}'.format(
                    self.current_epoch, self.current_step))

        # init meters for loss value and metric values
        loss_meter, metric_meters = self.get_init_meters()

        epoch_progress = self.get_progress_bar(
            iterable=self.test_loader,
            desc='Testing epoch {}'.format(self.current_epoch),
            remain_on_screen=False,
        )

        for idx, sample_batch in enumerate(epoch_progress):
            # sample_id is the index according to the dataset
            sample_id = self.test_loader.dataset.dataset.indices[idx]

            images = sample_batch[self.image_key].to(self.device)
            targets = sample_batch[self.target_key].to(self.device)
            masks = None if self.mask_key is None else sample_batch[
                self.mask_key].to(self.device)

            # model outputs inactivated logits
            logits = self.model(images)
            prob_maps = torch.sigmoid(logits)
            binary_maps = self.get_binary_maps(prob_maps)
            analytic_maps = self.get_analytic_maps(binary_maps, targets, masks)

            # criterion inputs inactivated logits
            current_loss = self.criterion(logits, targets)

            if torch.isnan(current_loss):
                LOGGER.warning('Criterion (%s) reached NaN during testing %d '
                               'sample at training epoch %d, training step %d',
                               self.configs.LOSS.LOSS_NAME, sample_id,
                               self.current_epoch, self.current_step)

            # update meters
            loss_meter.accumulate(current_loss.item())
            current_metrics = self.get_metrics(
                prob_maps, binary_maps, targets, masks)
            metric_meters.accumulate(current_metrics)

            self.save_figures(output_dir, sample_id, **{
                'images': images,
                'targets': targets,
                'masks': masks,
                'probability_maps': prob_maps,
                'binary_maps': binary_maps,
                'analytic_maps': analytic_maps,
            })

        epoch_progress.close()

        sample_ids = self.test_loader.dataset.dataset.indices
        self.save_metrics(output_dir, sample_ids, loss_meter, metric_meters)

    def run_finalizing(self):
        """Run finalizing process of agent."""

        # save last agent state and run testing on it
        last_ckpt_path = os.path.join(
            self.paths['ckpt_dir'], 'last' + self.configs.LOCAL.CKPT_EXT)
        self.save_agent_state(last_ckpt_path)
        last_ckpt_test_path = os.path.join(self.paths['out_dir'], 'last')
        self.make_sure_path_exist(last_ckpt_test_path)
        self.run_testing(last_ckpt_path, last_ckpt_test_path)

        for monitor_name in self.monitors.keys():
            ckpt_path = os.path.join(self.paths['ckpt_dir'],
                                     monitor_name + self.configs.LOCAL.CKPT_EXT)
            output_dir = os.path.join(self.paths['out_dir'], monitor_name)
            self.make_sure_path_exist(output_dir)
            self.run_testing(ckpt_path, output_dir)

        # close summary writer
        self.summ_writer.close()

    def get_binary_maps(self, prob_maps):
        """Get the threshold-ed output of probability maps."""
        thresh_mode = self.configs.METRICS.THRESHOLD.THRESHOLD_METHOD
        kwargs = {}
        kwargs['constant'] = self.configs.METRICS.THRESHOLD.CONSTANT
        kwargs['block_size'] = self.configs.METRICS.THRESHOLD.BLOCK_SIZE

        binary_maps = batch_thresholding(prob_maps, thresh_mode, **kwargs)
        binary_maps = torch.from_numpy(binary_maps)

        return binary_maps

    @staticmethod
    def get_analytic_maps(binary_maps, targets, masks=None):
        """Get analytic maps."""
        (true_pos, false_pos, _,
         false_neg) = binary_confusion.get_binary_confusion_matrix(
             binary_maps, targets, masks, reduction='none')

        analytics_maps = numpy.concatenate([false_pos, true_pos, false_neg],
                                           axis=1)

        analytics_maps = torch.from_numpy(analytics_maps)

        return analytics_maps

    # pylint: disable=arguments-differ
    def get_metrics(self, prob_maps, binary_maps, targets, masks=None):
        """Get metrics values on a batch of probability maps and targets."""
        metric_names = self.configs.METRICS.METRIC_NAMES
        return get_metrics(metric_names, prob_maps, binary_maps, targets, masks)

    def get_model(self):
        """Get model."""
        return get_model(self.configs)

    def get_criterion(self):
        """Get criterion."""
        return get_criterion(self.configs)

    def get_data_loaders(self):
        """Get data loaders."""
        return get_data_loader(self.configs)
