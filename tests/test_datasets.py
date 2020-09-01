"""Test custom datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from src.datasets.aria_dataset import ARIADataset, ARIAPILDataset
from src.datasets.chasedb1_dataset import CHASEDB1Dataset, CHASEDB1PILDataset
from src.datasets.drive_dataset import DRIVEDataset, DRIVEPILDataset
from src.datasets.hrf_dataset import HRFDataset, HRFPILDataset
from src.datasets.stare_dataset import STAREDataset, STAREPILDataset


def plot_sample(dataset, idx):
    """Plot sample of dataset given index."""
    def prepare_image(image):
        if isinstance(image, torch.Tensor):
            image = np.array(to_pil_image(image))

        elif isinstance(image, Image.Image):
            image = np.array(image)

        else:
            raise NotImplementedError

        return image

    # DO NOT access multiple time for different data of the same sample
    sample = dataset[idx]
    for key in dataset.sample_keys:
        locals()[key] = sample[key]
        locals()[key] = prepare_image(locals()[key])

    fig, axes = plt.subplots(1, len(dataset.sample_keys), figsize=(15, 5))
    for i in range(len(dataset.sample_keys)):
        shape = (locals()[dataset.sample_keys[i]]).shape
        if len(shape) == 2:
            axes[i].imshow(locals()[dataset.sample_keys[i]], cmap='gray')
        else:
            axes[i].imshow(locals()[dataset.sample_keys[i]])
        axes[i].set_title('{}, {}'.format(dataset.sample_keys[i], shape))

    return fig


class TestDatasets(unittest.TestCase):
    """Test custom datasets."""

    def test_aria_dataset(self):
        """Test ARIA dataset."""

        data_root = 'data'
        valid_ids = [0, 2]
        test_ids = [1, 3]

        for data_type, num_samples in {'amd': 23,
                                       'control': 61,
                                       'diabetic': 59}.items():

            aria_pil_dataset = ARIAPILDataset(data_root=data_root,
                                              data_type=data_type,
                                              download=True,
                                              extract=True)

            aria_train_dataset = ARIADataset(data_root=data_root,
                                             split_mode='train',
                                             valid_ids=valid_ids,
                                             test_ids=test_ids,
                                             preprocessing=None,
                                             augmentation=None,
                                             data_type=data_type)

            aria_valid_dataset = ARIADataset(data_root=data_root,
                                             split_mode='valid',
                                             valid_ids=valid_ids,
                                             test_ids=test_ids,
                                             preprocessing=None,
                                             augmentation=None,
                                             data_type=data_type)

            aria_test_dataset = ARIADataset(data_root=data_root,
                                            split_mode='test',
                                            valid_ids=valid_ids,
                                            test_ids=test_ids,
                                            preprocessing=None,
                                            augmentation=None,
                                            data_type=data_type)

            self.assertEqual(len(aria_pil_dataset), num_samples)

            train_num_sample = num_samples - len(valid_ids) - len(test_ids)
            self.assertEqual(len(aria_train_dataset), train_num_sample)

            train_sample_ids = list(range(num_samples))
            for idx in valid_ids + test_ids:
                train_sample_ids.remove(idx)

            self.assertEqual(aria_train_dataset.dataset.indices,
                             train_sample_ids)

            self.assertEqual(len(aria_valid_dataset), len(valid_ids))
            self.assertEqual(aria_valid_dataset.dataset.indices, valid_ids)

            self.assertEqual(len(aria_test_dataset), len(test_ids))
            self.assertEqual(aria_test_dataset.dataset.indices, test_ids)

            plot_sample(aria_train_dataset, 0).savefig(
                './tests/aria-{}-train.png'.format(data_type))
            plot_sample(aria_valid_dataset, 0).savefig(
                './tests/aria-{}-valid.png'.format(data_type))
            plot_sample(aria_test_dataset, 0).savefig(
                './tests/aria-{}-test.png'.format(data_type))

    def test_chasedb1_dataset(self):
        """test CHASEDB1 dataset."""

        num_samples = 28

        data_root = 'data'
        valid_ids = [0, 2]
        test_ids = [1, 3]

        chasedb1_pil_dataset = CHASEDB1PILDataset(data_root=data_root,
                                                  download=True,
                                                  extract=True)

        chasedb1_train_dataset = CHASEDB1Dataset(data_root=data_root,
                                                 split_mode='train',
                                                 valid_ids=valid_ids,
                                                 test_ids=test_ids,
                                                 preprocessing=None,
                                                 augmentation=None)

        chasedb1_valid_dataset = CHASEDB1Dataset(data_root=data_root,
                                                 split_mode='valid',
                                                 valid_ids=valid_ids,
                                                 test_ids=test_ids,
                                                 preprocessing=None,
                                                 augmentation=None)

        chasedb1_test_dataset = CHASEDB1Dataset(data_root=data_root,
                                                split_mode='test',
                                                valid_ids=valid_ids,
                                                test_ids=test_ids,
                                                preprocessing=None,
                                                augmentation=None)

        self.assertEqual(len(chasedb1_pil_dataset), num_samples)

        train_num_sample = num_samples - len(valid_ids) - len(test_ids)
        self.assertEqual(len(chasedb1_train_dataset), train_num_sample)

        train_sample_ids = list(range(num_samples))
        for idx in valid_ids + test_ids:
            train_sample_ids.remove(idx)

        self.assertEqual(chasedb1_train_dataset.dataset.indices,
                         train_sample_ids)

        self.assertEqual(len(chasedb1_valid_dataset), len(valid_ids))
        self.assertEqual(chasedb1_valid_dataset.dataset.indices, valid_ids)

        self.assertEqual(len(chasedb1_test_dataset), len(test_ids))
        self.assertEqual(chasedb1_test_dataset.dataset.indices, test_ids)

        plot_sample(chasedb1_train_dataset, 0).savefig(
            './tests/chase-train.png')
        plot_sample(chasedb1_valid_dataset, 0).savefig(
            './tests/chase-valid.png')
        plot_sample(chasedb1_test_dataset, 0).savefig(
            './tests/chase-test.png')

    def test_drive_dataset(self):
        """Test DRIVE dataset."""
        drive_train_pil_dataset = DRIVEPILDataset(data_root='data',
                                                  split_mode='train',
                                                  download_code=None,
                                                  download=True,
                                                  extract=True)
        drive_valid_pil_dataset = DRIVEPILDataset(data_root='data',
                                                  split_mode='valid',
                                                  download_code=None,
                                                  download=True,
                                                  extract=True)
        drive_test_pil_dataset = DRIVEPILDataset(data_root='data',
                                                 split_mode='test',
                                                 download_code=None,
                                                 download=True,
                                                 extract=True)

        drive_train_dataset = DRIVEDataset(data_root='data',
                                           split_mode='train',
                                           preprocessing=None,
                                           augmentation=None)
        drive_valid_dataset = DRIVEDataset(data_root='data',
                                           split_mode='valid',
                                           preprocessing=None,
                                           augmentation=None)
        drive_test_dataset = DRIVEDataset(data_root='data',
                                          split_mode='test',
                                          preprocessing=None,
                                          augmentation=None)

        self.assertEqual(len(drive_train_pil_dataset), 20)
        self.assertEqual(len(drive_valid_pil_dataset), 20)
        self.assertEqual(len(drive_test_pil_dataset), 20)
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)
        self.assertEqual(len(drive_test_dataset), 20)

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/drive-train.png')
        plot_sample(drive_valid_dataset, 0).savefig(
            './tests/drive-valid.png')
        plot_sample(drive_test_dataset, 0).savefig(
            './tests/drive-test.png')

    def test_hrf_dataset(self):
        """Test HRF dataset."""

        data_root = 'data'
        valid_ids = [0, 2]
        test_ids = [1, 3]

        for data_type, num_samples in {'all': 45,
                                       'healthy': 15,
                                       'glaucoma': 15,
                                       'diabetic': 15}.items():

            hrf_pil_dataset = HRFPILDataset(data_root=data_root,
                                            data_type=data_type,
                                            download=True,
                                            extract=True)

            hrf_train_dataset = HRFDataset(data_root=data_root,
                                           split_mode='train',
                                           valid_ids=valid_ids,
                                           test_ids=test_ids,
                                           preprocessing=None,
                                           augmentation=None,
                                           data_type=data_type)

            hrf_valid_dataset = HRFDataset(data_root=data_root,
                                           split_mode='valid',
                                           valid_ids=valid_ids,
                                           test_ids=test_ids,
                                           preprocessing=None,
                                           augmentation=None,
                                           data_type=data_type)

            hrf_test_dataset = HRFDataset(data_root=data_root,
                                          split_mode='test',
                                          valid_ids=valid_ids,
                                          test_ids=test_ids,
                                          preprocessing=None,
                                          augmentation=None,
                                          data_type=data_type)

            self.assertEqual(len(hrf_pil_dataset), num_samples)

            train_num_sample = num_samples - len(valid_ids) - len(test_ids)
            self.assertEqual(len(hrf_train_dataset), train_num_sample)

            train_sample_ids = list(range(num_samples))
            for idx in valid_ids + test_ids:
                train_sample_ids.remove(idx)

            self.assertEqual(hrf_train_dataset.dataset.indices,
                             train_sample_ids)

            self.assertEqual(len(hrf_valid_dataset), len(valid_ids))
            self.assertEqual(hrf_valid_dataset.dataset.indices, valid_ids)

            self.assertEqual(len(hrf_test_dataset), len(test_ids))
            self.assertEqual(hrf_test_dataset.dataset.indices, test_ids)

            plot_sample(hrf_train_dataset, 0).savefig(
                './tests/hrf-{}-train.png'.format(data_type))
            plot_sample(hrf_valid_dataset, 0).savefig(
                './tests/hrf-{}-valid.png'.format(data_type))
            plot_sample(hrf_test_dataset, 0).savefig(
                './tests/hrf-{}-test.png'.format(data_type))

    def test_stare_dataset(self):
        """Test STARE dataset."""

        num_samples = 20
        data_root = 'data'
        valid_ids = [0, 2]
        test_ids = [1, 3]

        stare_pil_dataset = STAREPILDataset(data_root=data_root,
                                            download=True,
                                            extract=True)

        stare_train_dataset = STAREDataset(data_root=data_root,
                                           split_mode='train',
                                           valid_ids=valid_ids,
                                           test_ids=test_ids,
                                           preprocessing=None,
                                           augmentation=None)

        stare_valid_dataset = STAREDataset(data_root=data_root,
                                           split_mode='valid',
                                           valid_ids=valid_ids,
                                           test_ids=test_ids,
                                           preprocessing=None,
                                           augmentation=None)

        stare_test_dataset = STAREDataset(data_root=data_root,
                                          split_mode='test',
                                          valid_ids=valid_ids,
                                          test_ids=test_ids,
                                          preprocessing=None,
                                          augmentation=None)

        self.assertEqual(len(stare_pil_dataset), num_samples)

        train_num_sample = num_samples - len(valid_ids) - len(test_ids)
        self.assertEqual(len(stare_train_dataset), train_num_sample)

        train_sample_ids = list(range(num_samples))
        for idx in valid_ids + test_ids:
            train_sample_ids.remove(idx)

        self.assertEqual(stare_train_dataset.dataset.indices,
                         train_sample_ids)

        self.assertEqual(len(stare_valid_dataset), len(valid_ids))
        self.assertEqual(stare_valid_dataset.dataset.indices, valid_ids)

        self.assertEqual(len(stare_test_dataset), len(test_ids))
        self.assertEqual(stare_test_dataset.dataset.indices, test_ids)

        plot_sample(stare_train_dataset, 0).savefig(
            './tests/stare-train.png')
        plot_sample(stare_valid_dataset, 0).savefig(
            './tests/stare-valid.png')
        plot_sample(stare_test_dataset, 0).savefig(
            './tests/stare-test.png')


if __name__ == "__main__":
    unittest.main()
