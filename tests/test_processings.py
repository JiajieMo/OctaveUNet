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

from src.datasets.drive_dataset import DRIVEDataset
from src.datasets.utils.get_dataset_statistics import get_channel_mean_std
from src.datasets.utils.check_data_integrity import check_binary_map

from src.processings.augmentations import random_adjust_brightness
from src.processings.augmentations import random_adjust_contrast
from src.processings.augmentations import random_adjust_gamma
from src.processings.augmentations import random_adjust_saturation
from src.processings.augmentations import random_affine_transform
from src.processings.augmentations import random_hflip
from src.processings.augmentations import random_vflip
from src.processings.augmentations import random_rotate

from src.processings.preprocessings import resize
from src.processings.preprocessings import normalization
from src.processings.preprocessings import vessel_enhancement

from src.processings.thresholdings import batch_thresholding


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


class TestAugmentations(unittest.TestCase):
    """Test custom agumentations, need visual examination to ensure."""

    def test_random_hflip(self):
        """Test random_hflip."""
        def random_hflip_with_kwargs(sample):
            """Update trigger probability of random_hflip for tesing."""
            kwagrs = {'trigger_prob': 1.0}  # default trigger_prob is 0.5
            sample = random_hflip(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/before-agumentations.png')

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=random_hflip_with_kwargs,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-hflip.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=None,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)

    def test_random_vflip(self):
        """Test random_vflip."""
        def random_vflip_with_kwargs(sample):
            """Update trigger probability of random_vflip for tesing."""
            kwagrs = {'trigger_prob': 1.0}  # default trigger_prob is 0.5
            sample = random_vflip(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/before-agumentations.png')

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=random_vflip_with_kwargs,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-vflip.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=None,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)

    def test_random_rotate(self):
        """Test random_rotate."""
        def random_rotate_with_kwargs(sample):
            """Update trigger probability of random_rotate for tesing."""
            kwagrs = {
                'trigger_prob': 1.0,  # default trigger_prob is 0.5
                'rotate_angle_range': (-180, 180),  # same as default range
            }
            sample = random_rotate(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/before-agumentations.png')

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=random_rotate_with_kwargs,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-rotate.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=None,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)

    def test_random_affine_transform(self):
        """Test random_affine_transform."""
        def random_affine_transform_with_kwargs(sample):
            """Update kwargs for tesing."""
            kwagrs = {
                'trigger_prob': 1.0,  # default trigger_prob is 0.5
                'rotate_angle_range': (-180, 180),  # same as default range
                'translate_range': (0, 0),  # same as default range
                'scale_range': (1, 1),  # same as default range
                'shear_range': (-5, 5),  # same as default range
            }
            sample = random_affine_transform(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/before-agumentations.png')

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=random_affine_transform_with_kwargs,
        )
        plot_sample(drive_train_dataset, 0).savefig(
            './tests/before-affine.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=None,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)

    def test_random_adjust_brightness(self):
        """Test random_adjust_brightness."""
        def random_adjust_brightness_with_kwargs(sample):
            """Update kwargs for tesing."""
            kwagrs = {
                'trigger_prob': 1.0,  # default trigger_prob is 0.5
                'brightness_factor_range': (0.8, 1.2),  # same as default range
            }
            sample = random_adjust_brightness(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/before-agumentations.png')

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=random_adjust_brightness_with_kwargs,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-adjust_brightness.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=None,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)

    def test_random_adjust_contrast(self):
        """Test random_adjust_contrast."""
        def random_adjust_contrast_with_kwargs(sample):
            """Update kwargs for tesing."""
            kwagrs = {
                'trigger_prob': 1.0,  # default trigger_prob is 0.5
                'contrast_factor_range': (0.8, 1.2),  # same as default range
            }
            sample = random_adjust_contrast(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/before-agumentations.png')

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=random_adjust_contrast_with_kwargs,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-adjust_contrast.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=None,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)

    def test_random_adjust_gamma(self):
        """Test random_adjust_gamma."""
        def random_adjust_gamma_with_kwargs(sample):
            """Update kwargs for tesing."""
            kwagrs = {
                'trigger_prob': 1.0,  # default trigger_prob is 0.5
                'gamma_range': (0.8, 1.2),  # same as default range
            }
            sample = random_adjust_gamma(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/before-agumentations.png')

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=random_adjust_gamma_with_kwargs,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-adjust_gamma.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=None,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)

    def test_random_adjust_saturation(self):
        """Test random_adjust_saturation."""
        def random_adjust_saturation_with_kwargs(sample):
            """Update kwargs for tesing."""
            kwagrs = {
                'trigger_prob': 1.0,  # default trigger_prob is 0.5
                'saturation_factor_range': (0.8, 1.2),  # same as default range
            }
            sample = random_adjust_saturation(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/before-agumentations.png')

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=random_adjust_saturation_with_kwargs,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-adjust_saturation.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=None,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)

    def test_multiple_augmentation(self):
        """Test using multiple augmentation at the same time."""
        # def chain_processes(processes, kwargses):
        #     def warpper_chain_processes(sample_data):
        #         for process, kwargs in zip(processes, kwargses):
        #             sample_data = process(sample_data, **kwargs)
        #         return sample_data
        #     return warpper_chain_processes

        # multiple_augmentation_kwargs = [
        #     {'trigger_prob': 1.0, 'brightness_factor_range': (0.9, 1.1)},
        #     {'trigger_prob': 1.0, 'contrast_factor_range': (0.9, 1.1)},
        #     {'trigger_prob': 1.0, 'gamma_range': (0.9, 1.1)},
        #     {'trigger_prob': 1.0, 'saturation_factor_range': (0.9, 1.1)},
        #     {'trigger_prob': 1.0, 'roate_angle_range': (-180, 180),
        #      'translate_range': (0, 0), 'scale_range': (1, 1),
        #      'shear_range': (-5, 5)},
        #     {'trigger_prob': 1.0},
        #     {'trigger_prob': 1.0},
        #     {'trigger_prob': 1.0, 'roate_angle_range': (-180, 180)},
        # ]

        def chain_processes(processes, kwargs):
            def warpper_chain_processes(sample_data):
                for process in processes:
                    sample_data = process(sample_data, **kwargs)
                return sample_data
            return warpper_chain_processes

        multiple_augmentation_kwargs = {
            'trigger_prob': 1.0,
            'brightness_factor_range': (0.9, 1.1),
            'contrast_factor_range': (0.9, 1.1),
            'gamma_range': (0.9, 1.1),
            'saturation_factor_range': (0.9, 1.1),
            'roate_angle_range': (-180, 180),
            'translate_range': (0, 0),
            'scale_range': (1, 1),
            'shear_range': (-5, 5),
        }

        multiple_augmentation = chain_processes(
            (random_adjust_brightness, random_adjust_contrast,
             random_adjust_gamma, random_adjust_saturation,
             random_hflip, random_vflip, random_rotate,
             random_affine_transform),
            multiple_augmentation_kwargs)

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/before-agumentations.png')

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=multiple_augmentation,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-agumentations.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=None,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)


class TestPreprocessings(unittest.TestCase):
    """Test custom preprocessing, need visual examination to ensure."""

    def test_resize(self):
        """Test resize."""

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=None,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/original.png')

        def resize_with_kwargs(sample):
            """Update kwargs for tesing."""
            kwagrs = {'size': (512, 512)}
            sample = resize(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=resize_with_kwargs,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-resize.png')

        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(drive_train_dataset[0]['image'].shape, (3, 512, 512))

    def test_normalization(self):
        """Test normalization."""
        channel_mean, channel_std = get_channel_mean_std(
            dataset=DRIVEDataset(data_root='data',
                                 split_mode='train',
                                 preprocessing=None,
                                 augmentation=None),
            image_key='image',
        )

        def normalization_with_kwargs(sample):
            """Update kwargs for tesing."""
            kwagrs = {
                'channel_mean': channel_mean,
                'channel_std': channel_std,
            }
            sample = normalization(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=normalization_with_kwargs,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-normalization.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=normalization_with_kwargs,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)

    def test_vessel_enhancement(self):
        """Test vessel enhancement."""

        def vessel_enhancement_with_kwargs(sample):
            """Update kwargs for tesing."""
            kwagrs = {'struture_elem_radius': 11}
            sample = vessel_enhancement(sample, **kwagrs)
            return sample

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=vessel_enhancement_with_kwargs,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-vessel-enhancement.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=vessel_enhancement_with_kwargs,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)

    def test_multiple_preprocessing(self):
        """Test multiple preprocessings used at the same time."""
        def chain_processes(processes, kwargses):
            def warpper_chain_processes(sample_data):
                for process, kwargs in zip(processes, kwargses):
                    sample_data = process(sample_data, **kwargs)
                return sample_data
            return warpper_chain_processes

        channel_mean, channel_std = get_channel_mean_std(
            dataset=DRIVEDataset(data_root='data',
                                 split_mode='train',
                                 preprocessing=None,
                                 augmentation=None),
            image_key='image',
        )

        multiple_preprocessing_kwargs = [
            {'channel_mean': channel_mean,
             'channel_std': channel_std},
            {'struture_elem_radius': 11},
        ]

        multiple_preprocessing = chain_processes(
            (normalization,
             vessel_enhancement),
            multiple_preprocessing_kwargs)

        drive_train_dataset = DRIVEDataset(
            data_root='data',
            split_mode='train',
            preprocessing=multiple_preprocessing,
            augmentation=None,
        )

        plot_sample(drive_train_dataset, 0).savefig(
            './tests/after-preprocessings.png')

        drive_valid_dataset = DRIVEDataset(
            data_root='data',
            split_mode='valid',
            preprocessing=multiple_preprocessing,
            augmentation=None,
        )

        # need extra visual examination of the samples
        self.assertEqual(len(drive_train_dataset), 20)
        self.assertEqual(len(drive_valid_dataset), 20)


class TestThresholdings(unittest.TestCase):
    """Test thresholding methods."""

    def test_constant_thresholding(self):
        """Test thresholding with constant."""

        # test batch of probability map with unsqueezed channel dimension
        prob_maps = torch.randn(2, 1, 10, 10)
        binary_maps = batch_thresholding(
            prob_maps, thresh_mode='constant', constant=0.5)

        self.assertEqual(binary_maps.shape, (2, 1, 10, 10))
        self.assertTrue(check_binary_map(binary_maps))

        # test batch of probability map with squeezed channel dimension
        prob_maps = torch.randn(2, 10, 10)
        binary_maps = batch_thresholding(
            prob_maps, thresh_mode='constant', constant=0.5)

        self.assertEqual(binary_maps.shape, (2, 1, 10, 10))
        self.assertTrue(check_binary_map(binary_maps))

        # test single sample of probability map with squeezed channel dimension
        prob_maps = torch.randn(1, 10, 10)
        binary_maps = batch_thresholding(
            prob_maps, thresh_mode='constant', constant=0.5)

        self.assertEqual(binary_maps.shape, (1, 1, 10, 10))
        self.assertTrue(check_binary_map(binary_maps))

    def test_skimage_thresholdings(self):
        """Test thresholding methods from skimage."""
        skimage_thresholding_methods = {
            'otsu': {'nbins': 256, 'return_all': False},
            'isodata': {'nbins': 256, 'return_all': False},
            'li': {'tolerance': None},
            'mean': {},
            'triangle': {'nbins': 256},
            'yen': {'nbins': 256},
            'niblack': {'window_size': 15, 'k': 0.2},
            'sauvola': {'window_size': 15, 'k': 0.2, 'r': None},
            'local': {'block_size': 11, 'method': 'gaussian', 'offset': 0,
                      'mode': 'reflect', 'param': None, 'cval': 0},
        }

        prob_maps = torch.randn(2, 1, 10, 10)

        for method, kwargs in skimage_thresholding_methods.items():
            binary_maps = batch_thresholding(
                prob_maps, thresh_mode=method, **kwargs)

            self.assertTrue(check_binary_map(binary_maps))


if __name__ == "__main__":
    unittest.main()
