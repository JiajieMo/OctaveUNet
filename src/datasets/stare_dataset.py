"""STARE dataset class implemented for PyTorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import gzip
import tarfile
import shutil
from os import remove
from os import path
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import Subset

from torchvision.transforms import functional as TF
from torchvision.datasets.utils import download_url, check_integrity, list_files

from src.datasets.utils.check_data_integrity import check_sample_ids

LOGGER = logging.getLogger(__name__)


class STAREDataset(Dataset):
    """STARE dataset with augmentation and formatting."""

    def __init__(self, data_root, split_mode, valid_ids, test_ids=None,
                 preprocessing=None, augmentation=None,
                 download=False, extract=False):
        self.sample_keys = ['image', 'target', 'target_aux']
        self.data_root = data_root
        self.valid_ids = valid_ids
        self.test_ids = test_ids if test_ids is not None else valid_ids
        self.preprocessing = preprocessing
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
                LOGGER.warning('Ignoring augmentation for valid set')

        elif self.split_mode == 'test':
            *_, self.dataset = self._get_subsets(download, extract)
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

    def _get_subsets(self, download, extract):
        pil_dataset = STAREPILDataset(data_root=self.data_root,
                                      download=download,
                                      extract=extract)
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


class STAREPILDataset(Dataset):
    """STARE dataset of original PIL images."""

    def __init__(self, data_root, download=False, extract=False):
        self.sample_keys = ['image', 'target', 'target_aux']
        self.data_root = path.expanduser(data_root)

        self._download_dir = path.join(self.data_root, 'STARE')
        self._tar_paths = [path.join(self._download_dir, 'stare-images.tar'),
                           path.join(self._download_dir, 'labels-ah.tar'),
                           path.join(self._download_dir, 'labels-vk.tar')]
        self._tar_md5s = ['d8b6f38d3a11c165d1f379a58e11c09b',
                          '6d679bfeb037b8e69f453b010eefeae1',
                          '060126ec6232ba24062144b48cfdfef1']
        self._image_dir = path.join(self._download_dir, 'stare-images')
        self._target_dir = path.join(self._download_dir, 'labels-ah')
        self._target_aux_dir = path.join(self._download_dir, 'labels-vk')

        if download:
            self._download()
        elif extract:
            self._extract()

        if not self._check_integrity():
            LOGGER.error('STARE dataset not found or corrupted')
            raise RuntimeError('STARE dataset not found or corrupted')

        self._image_paths = sorted(list_files(root=self._image_dir,
                                              suffix=('.ppm', '.PPM'),
                                              prefix=True))
        self._target_paths = sorted(list_files(root=self._target_dir,
                                               suffix=('.ppm', '.PPM'),
                                               prefix=True))
        self._target_aux_paths = sorted(list_files(root=self._target_aux_dir,
                                                   suffix=('.ppm', '.PPM'),
                                                   prefix=True))

        assert len(self._image_paths) == len(
            self._target_paths), 'STARE dataset corrupted'
        assert len(self._image_paths) == len(
            self._target_aux_paths), 'STARE dataset corrupted'

        LOGGER.debug('Retrieved all %d samples for STARE dataset',
                     len(self._image_paths))

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index], mode='r').convert('RGB')
        target = Image.open(self._target_paths[index], mode='r').convert('1')
        target_aux = Image.open(
            self._target_aux_paths[index], mode='r').convert('1')

        return {'image': image, 'target': target, 'target_aux': target_aux}

    def __len__(self):
        return len(self._image_paths)

    def _check_integrity(self):
        for tar_path, tar_md5 in zip(self._tar_paths, self._tar_md5s):
            if not check_integrity(tar_path, tar_md5):
                return False

        return True

    def _download(self):
        url_prefix = 'http://cecas.clemson.edu/~ahoover/stare/probing/'

        if not self._check_integrity():
            for tar_path, tar_md5 in zip(self._tar_paths, self._tar_md5s):
                url = url_prefix + path.basename(tar_path)
                download_url(url, root=self._download_dir, md5=tar_md5)
        else:
            LOGGER.debug('HRF dataset already downloaded and verified, '
                         'skipping download...')

        self._extract()

    def _extract(self):
        extract_dirs = [self._image_dir,
                        self._target_dir,
                        self._target_aux_dir]

        for i, tar_path in enumerate(self._tar_paths):
            with tarfile.TarFile(tar_path, 'r') as tar_file:
                tar_file.extractall(extract_dirs[i])

        for folder in extract_dirs:
            for gz_file_path in list_files(folder, suffix='.gz', prefix=True):
                extracted_file_path = gz_file_path[:-3]
                with gzip.open(gz_file_path, 'rb') as gz_file:
                    with open(extracted_file_path, 'wb') as extracted_file:
                        shutil.copyfileobj(gz_file, extracted_file)

                remove(gz_file_path)
