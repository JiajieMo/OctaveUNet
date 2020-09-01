"""Get data related objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.dataloader import DataLoader

from src.configs.config_node import ConfigNode

from src.processings.preprocessings import resize
from src.processings.preprocessings import normalization
from src.processings.preprocessings import vessel_enhancement

from src.processings.augmentations import random_adjust_brightness
from src.processings.augmentations import random_adjust_contrast
from src.processings.augmentations import random_adjust_gamma
from src.processings.augmentations import random_adjust_saturation
from src.processings.augmentations import random_hflip
from src.processings.augmentations import random_vflip
from src.processings.augmentations import random_rotate
from src.processings.augmentations import random_affine_transform

from src.datasets.aria_dataset import ARIAPILDataset, ARIADataset
from src.datasets.chasedb1_dataset import CHASEDB1PILDataset, CHASEDB1Dataset
from src.datasets.drive_dataset import DRIVEPILDataset, DRIVEDataset
from src.datasets.hrf_dataset import HRFPILDataset, HRFDataset
from src.datasets.stare_dataset import STAREPILDataset, STAREDataset

from src.datasets.utils.get_dataset_statistics import get_channel_mean_std

LOGGER = logging.getLogger(__name__)


def chain_processes(processes, kwargses):
    """Chain processes together."""
    def warpper_chain_processes(sample):
        for process, kwargs in zip(processes, kwargses):
            sample = process(sample, **kwargs)
        return sample
    return warpper_chain_processes


def get_preprocessing_pipeline(configs: ConfigNode):
    """Get preprocessing pipeline."""
    preprocessing_pipeline = configs.DATA.DATASET.PRE_PIPELINE
    if preprocessing_pipeline is not None and len(preprocessing_pipeline) > 0:
        processes = list()
        kwargses = list()
        for preprocess in preprocessing_pipeline:
            if preprocess == 'norm':
                channel_mean, channel_std = get_channel_mean_std(
                    dataset=get_pil_datasets(configs),
                    image_key=configs.DATA.DATASET.IMAGE_KEY,
                )

                kwargs = {'channel_mean': channel_mean,
                          'channel_std': channel_std}

                processes.append(normalization)
                kwargses.append(kwargs)

            elif preprocess == 'enhance':
                kwargs = {'selem_radius': 11}

                processes.append(vessel_enhancement)
                kwargses.append(kwargs)

            elif preprocess == 'resize':
                kwargs = {'size': (512, 512)}

                processes.append(resize)
                kwargses.append(kwargs)

            else:
                LOGGER.error('Invalid preprocessing: %s', preprocess)
                raise NotImplementedError

        preprocessings = chain_processes(processes, kwargses)

    else:
        preprocessings = None

    return preprocessings


def get_augmentation_pipeline(configs: ConfigNode):
    """Get augemntation pipeline."""
    augmentation_pipeline = configs.DATA.DATASET.AUG_PIPELINE
    if augmentation_pipeline is not None and len(augmentation_pipeline) > 0:
        augmentations = list()
        kwargses = list()

        for augmentation in augmentation_pipeline:
            if augmentation == 'brightness':
                augmentations.append(random_adjust_brightness)
                kwargses.append({})

            elif augmentation == 'contrast':
                augmentations.append(random_adjust_contrast)
                kwargses.append({})

            elif augmentation == 'gamma':
                augmentations.append(random_adjust_gamma)
                kwargses.append({})

            elif augmentation == 'saturation':
                augmentations.append(random_adjust_saturation)
                kwargses.append({})

            elif augmentation == 'hflip':
                augmentations.append(random_hflip)
                kwargses.append({})

            elif augmentation == 'vflip':
                augmentations.append(random_vflip)
                kwargses.append({})

            elif augmentation == 'rotate':
                augmentations.append(random_rotate)
                kwargses.append({})

            elif augmentation == 'affine':
                augmentations.append(random_affine_transform)
                kwargses.append({})

            else:
                LOGGER.error('Invalid augmentation: %s', augmentation)
                raise NotImplementedError

        augmentations = chain_processes(augmentations, kwargses)

    else:
        augmentations = None

    return augmentations


def get_pil_datasets(configs: ConfigNode):
    """Get datasets of PIL images."""
    dataset_name = configs.DATA.DATASET.DATASET_NAME
    data_root = configs.DATA.DATA_ROOT

    if dataset_name == 'aria':
        data_type = configs.DATA.DATASET.DATA_TYPE
        pil_dataset = ARIAPILDataset(
            data_root, data_type=data_type, download=False, extract=False)

    elif dataset_name == 'chase':
        pil_dataset = CHASEDB1PILDataset(
            data_root, download=False, extract=False)

    elif dataset_name == 'drive':
        train_pil_dataset = DRIVEPILDataset(data_root, split_mode='train',
                                            download_code=None, download=False,
                                            extract=False)

        valid_pil_dataset = DRIVEPILDataset(data_root, split_mode='valid',
                                            download_code=None, download=False,
                                            extract=False)

        pil_dataset = ConcatDataset([train_pil_dataset, valid_pil_dataset])

    elif dataset_name == 'stare':
        pil_dataset = STAREPILDataset(data_root, download=False, extract=False)

    elif dataset_name == 'hrf':
        data_type = configs.DATA.DATASET.DATA_TYPE
        pil_dataset = HRFPILDataset(data_root, data_type=data_type,
                                    download=False, extract=False)

    else:
        LOGGER.error('Invalid dataset_name: %s', dataset_name)
        raise NotImplementedError

    return pil_dataset


def get_datasets(configs: ConfigNode):
    """Get train, valid, and test datasets."""
    dataset_name = configs.DATA.DATASET.DATASET_NAME
    data_root = configs.DATA.DATA_ROOT

    preprocessing = get_preprocessing_pipeline(configs)
    augmentation = get_augmentation_pipeline(configs)

    if dataset_name == 'aria':
        valid_ids = configs.DATA.DATASET.VALID_IDS
        test_ids = configs.DATA.DATASET.TEST_IDS
        data_type = configs.DATA.DATASET.DATA_TYPE

        train_set = ARIADataset(data_root=data_root,
                                split_mode='train',
                                valid_ids=valid_ids,
                                test_ids=test_ids,
                                preprocessing=preprocessing,
                                augmentation=augmentation,
                                data_type=data_type,
                                download=False,
                                extract=False)

        valid_set = ARIADataset(data_root=data_root,
                                split_mode='valid',
                                valid_ids=valid_ids,
                                test_ids=test_ids,
                                preprocessing=preprocessing,
                                augmentation=None,
                                data_type=data_type,
                                download=False,
                                extract=False)

        test_set = ARIADataset(data_root=data_root,
                               split_mode='test',
                               valid_ids=valid_ids,
                               test_ids=test_ids,
                               preprocessing=preprocessing,
                               augmentation=None,
                               data_type=data_type,
                               download=False,
                               extract=False)

    elif dataset_name == 'chase':
        valid_ids = configs.DATA.DATASET.VALID_IDS
        test_ids = configs.DATA.DATASET.TEST_IDS

        train_set = CHASEDB1Dataset(data_root=data_root,
                                    split_mode='train',
                                    valid_ids=valid_ids,
                                    test_ids=test_ids,
                                    preprocessing=preprocessing,
                                    augmentation=augmentation,
                                    download=False,
                                    extract=False)

        valid_set = CHASEDB1Dataset(data_root=data_root,
                                    split_mode='valid',
                                    valid_ids=valid_ids,
                                    test_ids=test_ids,
                                    preprocessing=preprocessing,
                                    augmentation=None,
                                    download=False,
                                    extract=False)
        test_set = CHASEDB1Dataset(data_root=data_root,
                                   split_mode='test',
                                   valid_ids=valid_ids,
                                   test_ids=test_ids,
                                   preprocessing=preprocessing,
                                   augmentation=None,
                                   download=False,
                                   extract=False)

    elif dataset_name == 'drive':
        train_set = DRIVEDataset(data_root=data_root,
                                 split_mode='train',
                                 preprocessing=preprocessing,
                                 augmentation=augmentation,
                                 download_code=None,
                                 download=False,
                                 extract=False)

        valid_set = DRIVEDataset(data_root=data_root,
                                 split_mode='valid',
                                 preprocessing=preprocessing,
                                 augmentation=None,
                                 download_code=None,
                                 download=False,
                                 extract=False)

        test_set = DRIVEDataset(data_root=data_root,
                                split_mode='test',
                                preprocessing=preprocessing,
                                augmentation=None,
                                download_code=None,
                                download=False,
                                extract=False)

    elif dataset_name == 'stare':
        valid_ids = configs.DATA.DATASET.VALID_IDS
        test_ids = configs.DATA.DATASET.TEST_IDS

        train_set = STAREDataset(data_root=data_root,
                                 split_mode='train',
                                 valid_ids=valid_ids,
                                 test_ids=test_ids,
                                 preprocessing=preprocessing,
                                 augmentation=augmentation,
                                 download=False,
                                 extract=False)

        valid_set = STAREDataset(data_root=data_root,
                                 split_mode='valid',
                                 valid_ids=valid_ids,
                                 test_ids=test_ids,
                                 preprocessing=preprocessing,
                                 augmentation=None,
                                 download=False,
                                 extract=False)

        test_set = STAREDataset(data_root=data_root,
                                split_mode='test',
                                valid_ids=valid_ids,
                                test_ids=test_ids,
                                preprocessing=preprocessing,
                                augmentation=None,
                                download=False,
                                extract=False)

    elif dataset_name == 'hrf':
        valid_ids = configs.DATA.DATASET.VALID_IDS
        test_ids = configs.DATA.DATASET.TEST_IDS
        data_type = configs.DATA.DATASET.DATA_TYPE

        train_set = HRFDataset(data_root=data_root,
                               split_mode='train',
                               valid_ids=valid_ids,
                               test_ids=test_ids,
                               preprocessing=preprocessing,
                               augmentation=augmentation,
                               data_type=data_type,
                               download=False,
                               extract=False)

        valid_set = HRFDataset(data_root=data_root,
                               split_mode='valid',
                               valid_ids=valid_ids,
                               test_ids=test_ids,
                               preprocessing=preprocessing,
                               augmentation=None,
                               data_type=data_type,
                               download=False,
                               extract=False)

        test_set = HRFDataset(data_root=data_root,
                              split_mode='test',
                              valid_ids=valid_ids,
                              test_ids=test_ids,
                              preprocessing=preprocessing,
                              augmentation=None,
                              data_type=data_type,
                              download=False,
                              extract=False)

    else:
        LOGGER.error('Invalid dataset_name: %s', dataset_name)
        raise NotImplementedError(dataset_name)

    return train_set, valid_set, test_set


def get_data_loader(configs: ConfigNode):
    """Get data loader."""
    # get datasets
    train_set, valid_set, test_set = get_datasets(configs)

    train_batch_size = configs.DATA.TRAIN_BATCH_SIZE
    valid_batch_size = configs.DATA.VALID_BATCH_SIZE
    test_batch_size = 1

    shuffle = configs.DATA.ENABLE_SHUFFLE
    pin_memory = configs.DATA.ENABLE_PIN_MEMORY
    drop_last = configs.DATA.ENABLE_DROP_LAST

    train_loader = DataLoader(dataset=train_set,
                              batch_size=train_batch_size,
                              shuffle=shuffle,
                              sampler=None,
                              batch_sampler=None,
                              num_workers=0,
                              collate_fn=None,
                              pin_memory=pin_memory,
                              drop_last=drop_last,
                              timeout=0,
                              worker_init_fn=None,
                              multiprocessing_context=None)

    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=valid_batch_size,
                              shuffle=False,
                              sampler=None,
                              batch_sampler=None,
                              num_workers=0,
                              collate_fn=None,
                              pin_memory=pin_memory,
                              drop_last=drop_last,
                              timeout=0,
                              worker_init_fn=None,
                              multiprocessing_context=None)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=test_batch_size,
                             shuffle=False,
                             sampler=None,
                             batch_sampler=None,
                             num_workers=0,
                             collate_fn=None,
                             pin_memory=pin_memory,
                             drop_last=drop_last,
                             timeout=0,
                             worker_init_fn=None,
                             multiprocessing_context=None)

    dataset_name = configs.DATA.DATASET.DATASET_NAME
    LOGGER.debug('Retrieved dataset: %s', dataset_name)
    LOGGER.debug('Set batch size for training set: %d', train_batch_size)
    LOGGER.debug('Set batch size for validating set: %d', valid_batch_size)
    LOGGER.debug('Set batch size for testing set: %d', test_batch_size)

    return train_loader, valid_loader, test_loader


def get_sample_keys(configs: ConfigNode):
    """Get sample keys for reading data samples."""
    # get keys of data sample
    image_key = configs.DATA.DATASET.IMAGE_KEY
    target_key = configs.DATA.DATASET.TARGET_KEY
    # not all datasets have mask for region of interest
    try:
        mask_key = configs.DATA.DATASET.MASK_KEY
    except AttributeError:
        mask_key = None

    return image_key, target_key, mask_key
