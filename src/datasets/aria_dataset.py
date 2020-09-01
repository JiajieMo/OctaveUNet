"""ARIA dataset class implemented for PyTorch."""

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


class ARIADataset(Dataset):
    """ARIA dataset with augmentation and formatting."""

    def __init__(self, data_root, split_mode, valid_ids, test_ids=None,
                 preprocessing=None, augmentation=None,
                 data_type='amd', download=False, extract=False):
        self.sample_keys = ['image', 'target', 'target_aux']
        self.data_root = data_root
        self.preprocessing = preprocessing
        self.valid_ids = valid_ids
        self.test_ids = test_ids if test_ids is not None else valid_ids
        self.split_mode = split_mode
        self.data_type = data_type
        self.download = download
        self.extract = extract

        assert self.split_mode in ('train', 'valid', 'test')
        assert check_sample_ids(self.valid_ids)
        assert check_sample_ids(self.test_ids)

        LOGGER.debug('Split mode: %s', self.split_mode)
        LOGGER.debug('Sample ids for validating: %s', self.valid_ids)
        LOGGER.debug('Sample ids for testing: %s', self.test_ids)

        if self.split_mode == 'train':
            self.dataset, *_ = self._get_subsets()
            self.augmentation = augmentation

        elif self.split_mode == 'valid':
            _, self.dataset, _ = self._get_subsets()
            self.augmentation = None
            if augmentation is not None:
                LOGGER.warning('Ignoring augmentation for valid set')

        elif self.split_mode == 'test':
            *_, self.dataset = self._get_subsets()
            self.augmentation = None
            if augmentation is not None:
                LOGGER.warning('Ignoring augmentation for test set')

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

    def _get_subsets(self):
        pil_dataset = ARIAPILDataset(data_root=self.data_root,
                                     data_type=self.data_type,
                                     download=self.download,
                                     extract=self.extract)
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


class ARIAPILDataset(Dataset):
    """ARIA dataset of original PIL images."""

    def __init__(self, data_root, data_type='amd',
                 download=False, extract=False):
        self.sample_keys = ['image', 'target', 'target_aux']
        self.data_root = path.expanduser(data_root)
        self.data_type = data_type

        assert self.data_type in ('amd', 'control', 'diabetic')

        if self.data_type == 'amd':
            self._download_dir = path.join(self.data_root, 'ARIA/amd')
            self._zip_paths = [path.join(self._download_dir,
                                         'aria_a_markups.zip'),
                               path.join(self._download_dir,
                                         'aria_a_markup_vessel.zip')]
            self._zip_md5s = ['fc8d380d5570ca279522662a9657a6e5',
                              '2e42723f56f99b507037718e30f42b56']
            self._image_dir = path.join(self._download_dir, 'aria_a_markups')
            self._target_dir = path.join(self._download_dir,
                                         'aria_a_markup_vessel')

        elif self.data_type == 'control':
            self._download_dir = path.join(self.data_root, 'ARIA/control')
            self._zip_paths = [path.join(self._download_dir,
                                         'aria_c_markups.zip'),
                               path.join(self._download_dir,
                                         'aria_c_markup_vessel.zip'),
                               path.join(self._download_dir,
                                         'aria_c_markupdiscfovea.zip')]
            self._zip_md5s = ['047904ea1c2b2cf8e74a63f22d02727d',
                              'b5f378cd644387be84f5f4a972e9107e',
                              '3ec8eb2ab3b690b6eaf331702262f742']
            self._image_dir = path.join(self._download_dir, 'aria_c_markups')
            self._target_dir = path.join(self._download_dir,
                                         'aria_c_markup_vessel')
            self._mask_dir = path.join(self._download_dir,
                                       'aria_c_markupdiscfovea')

        elif self.data_type == 'diabetic':
            self._download_dir = path.join(self.data_root, 'ARIA/diabetic')
            self._zip_paths = [path.join(self._download_dir,
                                         'aria_d_markups.zip'),
                               path.join(self._download_dir,
                                         'aria_d_markup_vessel.zip'),
                               path.join(self._download_dir,
                                         'aria_d_markupdiscfovea.zip')]
            self._zip_md5s = ['b7d8d6f873e2c0ee7b453cfb098171a1',
                              '21971de87668e5e6cd65a66324d70183',
                              '1c302848a665a05be426bef4b4c17a78']
            self._image_dir = path.join(self._download_dir,
                                        'aria_d_markups')
            self._target_dir = path.join(self._download_dir,
                                         'aria_d_markup_vessel')
            self._mask_dir = path.join(self._download_dir,
                                       'aria_d_markupdiscfovea')

        else:
            LOGGER.error('Data_type must be either "amd", "control", '
                         'or "diabetic", instead got %s', self.data_type)
            raise NotImplementedError('Data_type must be either "amd", '
                                      '"control", or "diabetic", instead '
                                      'got {}'.format(self.data_type))

        if download:
            self._download()
        elif extract:
            self._extract()

        if not self._check_integrity():
            LOGGER.error('ARIA dataset not found or corrupted')
            raise RuntimeError('ARIA dataset not found or corrupted')

        self._image_paths = sorted(list_files(root=self._image_dir,
                                              suffix=('.tif', '.TIF'),
                                              prefix=True))

        self._target_paths = sorted([
            target_path for target_path in list_files(
                root=self._target_dir,
                suffix=('.tif', '.TIF'),
                prefix=True
            ) if target_path.split('/')[-1].split('.')[0].endswith('BSS')
        ])

        self._target_aux_paths = sorted([
            target_path for target_path in list_files(
                root=self._target_dir,
                suffix=('.tif', '.TIF'),
                prefix=True
            ) if target_path.split('/')[-1].split('.')[0].endswith('BDP')
        ])

        assert len(self._image_paths) == len(
            self._target_paths), 'ARIA dataset corrupted'
        assert len(self._image_paths) == len(
            self._target_aux_paths), 'ARIA dataset corrupted'

        if self.data_type != 'amd':
            self._mask_paths = sorted(list_files(root=self._mask_dir,
                                                 suffix=('.tif', '.TIF'),
                                                 prefix=True))
            assert len(self._image_paths) == len(
                self._mask_paths), 'ARIA dataset corrupted'

        LOGGER.debug('Retrieved all %d samples for ARIA dataset',
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

    def _check_integrity(self):
        for zip_path, zip_md5 in zip(self._zip_paths, self._zip_md5s):
            if not check_integrity(zip_path, zip_md5):
                return False

        return True

    def _download(self):
        url_prefix = 'http://pcwww.liv.ac.uk/~yzheng/aria/'

        if not self._check_integrity():
            for zip_path, zip_md5 in zip(self._zip_paths, self._zip_md5s):
                url = url_prefix + path.basename(zip_path)
                download_url(url, root=self._download_dir, md5=zip_md5)
        else:
            LOGGER.debug('ARIA dataset already downloaded and verified, '
                         'skipping download')

        self._extract()

    def _extract(self):
        extract_dirs = self._download_dir

        for zip_path in self._zip_paths:
            with ZipFile(zip_path, 'r') as zip_file:
                zip_file.extractall(extract_dirs)
