"""DRIVE dataset class implemented for PyTorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from zipfile import ZipFile

from os import path
from PIL import Image

from torch.utils.data import Dataset

from torchvision.transforms import functional as TF
from torchvision.datasets.utils import download_url, check_integrity, list_files


LOGGER = logging.getLogger(__name__)


class DRIVEDataset(Dataset):
    """DRIVE dataset with augmentation and formatting."""

    def __init__(self, data_root, split_mode,
                 preprocessing=None, augmentation=None,
                 download_code=None, download=False, extract=False):
        self.data_root = data_root
        self.preprocessing = preprocessing
        self.split_mode = split_mode

        if self.split_mode == 'train':
            self.sample_keys = ['image', 'target', 'mask']
            self.augmentation = augmentation

        elif self.split_mode in ('valid', 'test'):
            self.sample_keys = ['image', 'target', 'mask', 'target_aux']
            self.augmentation = None
            if augmentation is not None:
                LOGGER.warning('Ignoring augmentation for %s set',
                               self.split_mode)

        else:
            LOGGER.error('Invalid split mode: %s', self.split_mode)
            raise NotImplementedError('Invalid split mode: {}'.format(
                self.split_mode))

        LOGGER.debug('Split mode: %s', self.split_mode)

        self.dataset = DRIVEPILDataset(data_root=self.data_root,
                                       split_mode=self.split_mode,
                                       download_code=download_code,
                                       download=download,
                                       extract=extract)

    def __getitem__(self, index):
        sample = self.dataset[index]

        if self.preprocessing is not None:
            sample = self.preprocessing(sample)

        if self.augmentation is not None:
            sample = self.augmentation(sample)

        return {key: TF.to_tensor(value) for key, value in sample.items()}

    def __len__(self):
        return len(self.dataset)


class DRIVEPILDataset(Dataset):
    """DRIVE dataset of original PIL images."""

    def __init__(self, data_root, split_mode,
                 download_code=None, download=False, extract=False):
        self.data_root = path.expanduser(data_root)
        self.split_mode = split_mode

        self._zip_path = path.join(self.data_root, 'DRIVE', 'DRIVE.zip')
        self._zip_md5 = 'a91f25272507b1f53132d03a69030de8'

        if self.split_mode == 'train':
            self._image_dir = path.join(self.data_root, 'DRIVE',
                                        'training', 'images')
            self._target_dir = path.join(self.data_root, 'DRIVE',
                                         'training', '1st_manual')
            self._mask_dir = path.join(self.data_root, 'DRIVE',
                                       'training', 'mask')
            self.sample_keys = ['image', 'target', 'mask']

        elif self.split_mode in ('valid', 'test'):
            self._image_dir = path.join(self.data_root, 'DRIVE',
                                        'test', 'images')
            self._target_dir = path.join(self.data_root, 'DRIVE',
                                         'test', '1st_manual')
            self._mask_dir = path.join(self.data_root, 'DRIVE',
                                       'test', 'mask')
            self._target_aux_dir = path.join(self.data_root, 'DRIVE',
                                             'test', '2nd_manual')
            self.sample_keys = ['image', 'target', 'mask', 'target_aux']

        else:
            LOGGER.error('Invalid split mode: %s', self.split_mode)
            raise NotImplementedError('Invalid split mode: {}'.format(
                self.split_mode))

        if download:
            self._download(download_code)
        elif extract:
            self._extract()

        if not check_integrity(self._zip_path, self._zip_md5):
            LOGGER.error('DRIVE dataset not found or corrupted')
            raise RuntimeError('DRIVE dataset not found or corrupted')

        self._image_paths = sorted(list_files(root=self._image_dir,
                                              suffix=('.tif', '.TIF'),
                                              prefix=True))
        self._target_paths = sorted(list_files(root=self._target_dir,
                                               suffix=('.gif', '.GIF'),
                                               prefix=True))
        self._mask_paths = sorted(list_files(root=self._mask_dir,
                                             suffix=('.gif', '.GIF'),
                                             prefix=True))
        if self.split_mode in ('valid', 'test'):
            self._target_aux_paths = sorted(list_files(
                root=self._target_aux_dir, suffix=('.gif', '.GIF'), prefix=True))
            assert len(self._image_paths) == len(
                self._target_aux_paths), 'DRIVE dataset corrupted'

        assert len(self._image_paths) == len(
            self._target_paths), 'DRIVE dataset corrupted'
        assert len(self._image_paths) == len(
            self._mask_paths), 'DRIVE dataset corrupted'

        LOGGER.debug('Retrieved all %d samples for DRIVE dataset',
                     len(self._image_paths))

        # for compatibility of other datasets
        self.indices = list(range(len(self._image_paths)))

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index], mode='r').convert('RGB')
        target = Image.open(self._target_paths[index], mode='r').convert('1')
        mask = Image.open(self._mask_paths[index], mode='r').convert('1')

        if self.split_mode in ('valid', 'test'):
            target_aux = Image.open(self._target_paths[index],
                                    mode='r').convert('1')
            return {'image': image, 'target': target,
                    'mask': mask, 'target_aux': target_aux}

        return {'image': image, 'target': target, 'mask': mask}

    def __len__(self):
        return len(self._image_paths)

    def _download(self, usercode):
        download_dir = path.join(self.data_root, 'DRIVE')
        zip_url_prefix = 'http://www.isi.uu.nl/Research/Databases/DRIVE/free.php?usercode='

        if not check_integrity(self._zip_path, md5=self._zip_md5):
            if usercode is not None:
                zip_url = zip_url_prefix + str(usercode)
                download_url(zip_url, root=download_dir, md5=self._zip_md5)
                self._extract()
            else:
                LOGGER.debug('Skipping download because invalid download code')
        else:
            LOGGER.debug('DRIVE already downloaded and verified')

        self._extract()

    def _extract(self):
        extract_dir = self.data_root

        with ZipFile(self._zip_path, 'r') as zip_file:
            zip_file.extractall(extract_dir)
