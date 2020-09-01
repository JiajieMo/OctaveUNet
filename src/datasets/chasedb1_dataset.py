"""CHASEDB1 dataset class implemented for PyTorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from zipfile import ZipFile

from os import path
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import Subset

from torchvision.transforms import functional as TF
from torchvision.datasets.utils import download_url, check_integrity, list_files

from src.datasets.utils.check_data_integrity import check_sample_ids

LOGGER = logging.getLogger(__name__)


class CHASEDB1Dataset(Dataset):
    """CHASEDB1 dataset with augmentation and formatting."""

    def __init__(self, data_root, split_mode, valid_ids, test_ids=None,
                 preprocessing=None, augmentation=None,
                 download=False, extract=False):
        self.sample_keys = ['image', 'target', 'target_aux']
        self.data_root = data_root
        self.preprocessing = preprocessing
        self.valid_ids = valid_ids
        self.test_ids = test_ids if test_ids is not None else valid_ids
        self.split_mode = split_mode

        assert self.split_mode in ('train', 'valid', 'test')
        assert check_sample_ids(self.valid_ids)
        assert check_sample_ids(self.test_ids)

        LOGGER.debug('Split mode: %s', self.split_mode)
        LOGGER.debug('Sample ids for validating: %s', self.valid_ids)
        LOGGER.debug('Sample ids for testing: %s', self.test_ids)

        if self.split_mode == 'train':
            self.dataset, *_ = self._get_subsets(download, extract)
            self.augmentation = augmentation

        elif self.split_mode == 'valid':
            _, self.dataset, _ = self._get_subsets(download, extract)
            self.augmentation = None
            if augmentation is not None:
                LOGGER.debug('Ignoring augmentation for valid set')

        elif self.split_mode == 'test':
            *_, self.dataset = self._get_subsets(download, extract)
            self.augmentation = None
            if augmentation is not None:
                LOGGER.debug('Ignoring augmentation for test set')

        else:
            LOGGER.error('Invalid split mode: %s', self.split_mode)
            raise NotImplementedError('Invalid split mode: {}'.format(
                                      self.split_mode))

    def __getitem__(self, index):
        sample = self.dataset[index]

        if self.preprocessing is not None:
            sample = self.preprocessing(sample)

        if self.augmentation is not None:
            sample = self.augmentation(sample)

        return {key: TF.to_tensor(value) for key, value in sample.items()}

    def __len__(self):
        return len(self.dataset)

    def _get_subsets(self, download, extract):
        pil_dataset = CHASEDB1PILDataset(data_root=self.data_root,
                                         download=download, extract=extract)
        num_samples = len(pil_dataset)

        try:
            train_ids = [idx for idx in range(num_samples) if (
                idx not in self.valid_ids) and (idx not in self.test_ids)]

            valid_ids = [idx for idx in range(
                num_samples) if idx in self.valid_ids]

            test_ids = [idx for idx in range(
                num_samples) if idx in self.test_ids]

            train_set = Subset(pil_dataset, train_ids)
            valid_set = Subset(pil_dataset, valid_ids)
            test_set = Subset(pil_dataset, test_ids)

        except AttributeError as error:
            LOGGER.error(error)
            LOGGER.error('Invalid sample ids for valid: %s, or for test: %s',
                         self.valid_ids, self.test_ids)
            raise AttributeError('Invalid sample ids for valid: {}, or for '
                                 'test: {}'.format(self.valid_ids,
                                                   self.test_ids))

        return train_set, valid_set, test_set


class CHASEDB1PILDataset(Dataset):
    """CHASEDB1 dataset of original PIL images."""

    def __init__(self, data_root, download=False, extract=False):
        self.sample_keys = ['image', 'target', 'target_aux']
        self.data_root = path.expanduser(data_root)

        self._zip_path = path.join(self.data_root, 'CHASEDB1', 'CHASEDB1.zip')
        self._zip_md5 = 'd9e47c4bac125b29996fae4380a68db1'

        self._image_dir = path.join(self.data_root, 'CHASEDB1', 'CHASEDB1')
        self._target_dir = self._image_dir

        if download:
            self._download()
        elif extract:
            self._extract()

        if not check_integrity(self._zip_path, self._zip_md5):
            LOGGER.error('CHASEDB1 dataset not found or corrupted')
            raise RuntimeError('CHASEDB1 dataset not found or corrupted')

        self._image_paths = sorted(list_files(root=self._image_dir,
                                              suffix=('.jpg', '.JPG'),
                                              prefix=True))

        self._target_paths = sorted([
            target_path for target_path in list_files(
                root=self._target_dir,
                suffix=('.png', '.PNG'),
                prefix=True
            ) if target_path.split('/')[-1].split('.')[0].endswith('1stHO')
        ])

        self._target_aux_paths = sorted([
            target_path for target_path in list_files(
                root=self._target_dir,
                suffix=('.png', '.PNG'),
                prefix=True
            ) if target_path.split('/')[-1].split('.')[0].endswith('2ndHO')
        ])

        assert len(self._image_paths) == len(
            self._target_paths), 'CHASEDB1 dataset corrupted'
        assert len(self._image_paths) == len(
            self._target_aux_paths), 'CHASEDB1 dataset corrupted'

        LOGGER.debug('Retrieved all %d samples for CHASEDB1 dataset',
                     len(self._image_paths))

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index], mode='r').convert('RGB')
        target = Image.open(self._target_paths[index], mode='r').convert('1')
        target_aux = Image.open(
            self._target_aux_paths[index], mode='r').convert('1')

        return {'image': image,
                'target': target,
                'target_aux': target_aux}

    def __len__(self):
        return len(self._image_paths)

    def _download(self):
        download_dir = path.join(self.data_root, 'CHASEDB1')
        zip_url = 'https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip'

        if not check_integrity(self._zip_path, md5=self._zip_md5):
            download_url(zip_url, root=download_dir, md5=self._zip_md5)
            self._extract()
        else:
            LOGGER.debug('CHASEDB1 dataset already downloaded and verified, '
                         'skipping download')
        self._extract()

    def _extract(self):
        extract_dir = path.join(self.data_root, 'CHASEDB1', 'CHASEDB1')

        with ZipFile(self._zip_path, 'r') as zip_file:
            zip_file.extractall(extract_dir)
