"""HRF dataset class implemented for PyTorch."""

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


class HRFDataset(Dataset):
    """HRF dataset with augmentation and formatting."""

    def __init__(self, data_root, split_mode, valid_ids, test_ids=None,
                 preprocessing=None, augmentation=None,
                 data_type='all', download=False, extract=False):
        self.sample_keys = ['image', 'target', 'mask']
        self.data_root = data_root
        self.preprocessing = preprocessing
        self.valid_ids = valid_ids
        self.test_ids = test_ids if test_ids is not None else valid_ids
        self.split_mode = split_mode
        self.data_type = data_type

        assert self.data_type in ('all', 'healthy', 'glaucoma', 'diabetic')
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
        pil_dataset = HRFPILDataset(data_root=self.data_root,
                                    data_type=self.data_type,
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


class HRFPILDataset(Dataset):
    """HRF dataset of original PIL images."""

    def __init__(self, data_root, data_type='all', download=False,
                 extract=False):
        self.sample_keys = ['image', 'target', 'mask']
        self.data_root = path.expanduser(data_root)
        self.data_type = data_type
        assert self.data_type in ('all', 'healthy', 'glaucoma', 'diabetic')

        if self.data_type == 'all':
            self._download_dir = path.join(self.data_root, 'HRF', 'all_data')
            self._zip_paths = [path.join(self._download_dir, 'all.zip'), ]
            self._zip_md5s = ['4587b74f4101e7456364b9b24d725576', ]
            self._image_dir = path.join(self._download_dir, 'images')
            self._target_dir = path.join(self._download_dir, 'manual1')
            self._mask_dir = path.join(self._download_dir, 'mask')

        elif self.data_type == 'healthy':
            self._download_dir = path.join(self.data_root, 'HRF',
                                           'healthy_data')
            self._zip_paths = [path.join(self._download_dir, 'healthy.zip'),
                               path.join(self._download_dir,
                                         'healthy_manualsegm.zip'),
                               path.join(self._download_dir,
                                         'healthy_fovmask.zip')]
            self._zip_md5s = ['4d65eba0898433c9e82f81d9c239e6cf',
                              'caefa3ea2351150df9c015985bedc543',
                              '25440d8cb1ba37ad84eb93df1b1d284e']
            self._image_dir = path.join(self._download_dir, 'healthy')
            self._target_dir = path.join(self._download_dir,
                                         'healthy_manualsegm')
            self._mask_dir = path.join(self._download_dir, 'healthy_fovmask')

        elif self.data_type == 'glaucoma':
            self._download_dir = path.join(self.data_root, 'HRF',
                                           'glaucoma_data')
            self._zip_paths = [path.join(self._download_dir, 'glaucoma.zip'),
                               path.join(self._download_dir,
                                         'glaucoma_manualsegm.zip'),
                               path.join(self._download_dir,
                                         'glaucoma_fovmask.zip')]
            self._zip_md5s = ['ce6c3749c35da746af70dd892b9d13ff',
                              '684b0fdedb157f67ece711062c798f2e',
                              '18edea6aee1326712f2cf1fd6b48ed51']
            self._image_dir = path.join(self._download_dir, 'glaucoma')
            self._target_dir = path.join(self._download_dir,
                                         'glaucoma_manualsegm')
            self._mask_dir = path.join(self._download_dir, 'glaucoma_fovmask')

        elif self.data_type == 'diabetic':
            self._download_dir = path.join(self.data_root, 'HRF',
                                           'diabetic_data')
            self._zip_paths = [path.join(self._download_dir,
                                         'diabetic_retinopathy.zip'),
                               path.join(self._download_dir,
                                         'diabetic_retinopathy_manualsegm.zip'),
                               path.join(self._download_dir,
                                         'diabetic_retinopathy_fovmask.zip')]
            self._zip_md5s = ['dfe216e29a865c3c8a52767757ca9430',
                              '808488d13ce516c7cf3f2d1204d31065',
                              'c197a6beea0c28d3317295f569ebc4ce']
            self._image_dir = path.join(self._download_dir,
                                        'diabetic_retinopathy')
            self._target_dir = path.join(self._download_dir,
                                         'diabetic_retinopathy_manualsegm')
            self._mask_dir = path.join(self._download_dir,
                                       'diabetic_retinopathy_fovmask')

        else:
            LOGGER.error('Data_type must be either "all", "healthy", '
                         '"glaucoma" or "diabetic"')
            raise NotImplementedError('Data_type must be either "all", '
                                      '"healthy", "glaucoma" or "diabetic"')

        if download:
            self._download()
        elif extract:
            self._extract()

        if not self._check_integrity():
            LOGGER.error('HRF dataset not found or corrupted')
            raise RuntimeError('HRF dataset not found or corrupted')

        self._image_paths = sorted(list_files(root=self._image_dir,
                                              suffix=('.jpg', '.JPG'),
                                              prefix=True))
        self._target_paths = sorted(list_files(root=self._target_dir,
                                               suffix=('.tif', '.TIF'),
                                               prefix=True))
        self._mask_paths = sorted(list_files(root=self._mask_dir,
                                             suffix=('.tif', '.TIF'),
                                             prefix=True))

        assert len(self._image_paths) == len(
            self._target_paths), 'HRF dataset corrupted'
        assert len(self._image_paths) == len(
            self._mask_paths), 'HRF dataset corrupted'

        LOGGER.debug('Retrieved all %d samples for HRF dataset',
                     len(self._image_paths))

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index], mode='r').convert('RGB')
        target = Image.open(self._target_paths[index], mode='r').convert('1')
        mask = Image.open(self._mask_paths[index], mode='r').convert('1')

        return {'image': image, 'target': target, 'mask': mask}

    def __len__(self):
        return len(self._image_paths)

    def _check_integrity(self):
        for zip_path, zip_md5 in zip(self._zip_paths, self._zip_md5s):
            if not check_integrity(zip_path, zip_md5):
                return False

        return True

    def _download(self):
        url_prefix = 'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/'

        if not self._check_integrity():
            for zip_path, zip_md5 in zip(self._zip_paths, self._zip_md5s):
                url = url_prefix + path.basename(zip_path)
                download_url(url, root=self._download_dir, md5=zip_md5)
        else:
            LOGGER.debug('HRF dataset already downloaded and verified, '
                         'skipping download...')

        self._extract()

    def _extract(self):
        if self.data_type == 'all':
            extract_dirs = [self._download_dir, ]
        else:
            extract_dirs = [self._image_dir, self._target_dir, self._mask_dir]

        for i, zip_path in enumerate(self._zip_paths):
            with ZipFile(zip_path, 'r') as zip_file:
                zip_file.extractall(extract_dirs[i])
