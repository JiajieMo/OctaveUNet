"""Test config_node module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import tempfile

from src.configs.config_node import ConfigNode


def get_dummy_config():
    """Get dummy config."""
    configs = ConfigNode()
    configs.DEVICE = ConfigNode()
    configs.DEVICE.ENABLE_CUDA = True
    configs.DEVICE.DEVICE_IDS = [0, 1]
    configs.MODEL = ConfigNode()
    configs.MODEL.MODEL_NAME = 'ANN'

    configs.DUMMY_FOR_TEST_STR = ConfigNode()
    configs.DUMMY_FOR_TEST_STR.KEY1 = 1
    configs.DUMMY_FOR_TEST_STR.KEY2 = 2
    configs.DUMMY_FOR_TEST_STR.FOO = ConfigNode()
    configs.DUMMY_FOR_TEST_STR.FOO.KEY1 = 1
    configs.DUMMY_FOR_TEST_STR.FOO.KEY2 = 2
    configs.DUMMY_FOR_TEST_STR.FOO.BAR = ConfigNode()
    configs.DUMMY_FOR_TEST_STR.FOO.BAR.KEY1 = 1
    configs.DUMMY_FOR_TEST_STR.FOO.BAR.KEY2 = 2

    configs.LOCAL = ConfigNode()
    configs.register_deprecated_key('DEPRECATED_KEY1')
    configs.register_deprecated_key('LOCAL.DEPRECATED_KEY2')

    configs.register_renamed_key(
        'EXAMPLE.OLD.KEY',
        'EXAMPLE.NEW.KEY',
        message='Please update your config fil config file.',
    )

    configs.NEW_ALLOWED = ConfigNode(new_allowed=True)
    return configs


class TestConfigNode(unittest.TestCase):
    """Test ConfigNode."""

    def test_config_node_immutability(self):
        """Test immutability of config_node."""
        config_node = ConfigNode()
        config_node.foo = 0
        config_node.bar = 'bar'
        config_node.freeze()

        with self.assertRaises(AttributeError):
            config_node.foo = 1
            config_node.bar = 'blah'
        self.assertTrue(config_node.is_frozen())
        self.assertTrue(config_node.foo == 0)

        config_node.defrost()
        self.assertFalse(config_node.is_frozen())
        config_node.foo = 1
        config_node.bar = 'blah'
        self.assertEqual(config_node.foo, 1)
        self.assertEqual(config_node.bar, 'blah')

        config_node = ConfigNode()
        config_node.foo = 0
        config_node.bar = 'bar'
        config_node.second_level = ConfigNode()
        config_node.second_level.foo = 0
        config_node.second_level.bar = 'bar'
        config_node.freeze()
        self.assertTrue(config_node.is_frozen())
        with self.assertRaises(AttributeError):
            config_node.second_level.foo = 1
            config_node.second_level.bar = 'blah'
        self.assertEqual(config_node.second_level.foo, 0)

    def test_new_allowed(self):
        """Test new allowed config node."""
        configs = ConfigNode()
        configs.DEVICE = ConfigNode()

        configs_new_allowed = ConfigNode()
        configs_new_allowed.DEVICE = ConfigNode(new_allowed=True)

        configs_2 = ConfigNode()
        configs_2.DEVICE = ConfigNode()
        configs_2.DEVICE.ENABLE_CUDA = True
        configs_2.DEVICE.DEVICE_IDS = [0, 1]

        with tempfile.NamedTemporaryFile('wt') as configs_2_file:
            configs_2_file.write(configs_2.dump())
            configs_2_file.flush()

            configs_new_allowed.merge_from_file(configs_2_file.name)
            self.assertTrue(configs_new_allowed.DEVICE.ENABLE_CUDA)
            self.assertEqual(configs_new_allowed.DEVICE.DEVICE_IDS, [0, 1])

            with self.assertRaises(KeyError):
                configs.merge_from_file(configs_2_file.name)


class TestConfigsUseCase(unittest.TestCase):
    """Test configutation use cases."""

    def test_copy_configs(self):
        """Test copying configs."""
        configs = get_dummy_config()
        configs_2 = configs.clone()

        model_type = configs.MODEL.MODEL_NAME
        configs_2.MODEL.MODEL_NAME = 'SVM'

        self.assertEqual(configs.MODEL.MODEL_NAME, model_type)
        self.assertNotEqual(configs.MODEL.MODEL_NAME,
                            configs_2.MODEL.MODEL_NAME)

    def test_merge_configs(self):
        """Test merging configs from other configs."""
        configs = get_dummy_config()
        configs_2 = configs.clone()

        model_type = configs.MODEL.MODEL_NAME
        configs_2.MODEL.MODEL_NAME = 'SVM'

        configs.merge_from_other_config(configs_2)
        self.assertEqual(configs.MODEL.MODEL_NAME,
                         configs_2.MODEL.MODEL_NAME)
        self.assertNotEqual(configs.MODEL.MODEL_NAME, model_type)

        # test merging configs from other configs with valid keys
        configs = get_dummy_config()
        configs_2 = ConfigNode()
        configs_2.DEVICE = ConfigNode()
        configs_2.DEVICE.ENABLE_CUDA = False
        configs_2.MODEL = ConfigNode()
        configs_2.MODEL.MODEL_NAME = 'SVM'

        configs.merge_from_other_config(configs_2)

        self.assertEqual(configs.DEVICE.ENABLE_CUDA,
                         configs_2.DEVICE.ENABLE_CUDA)
        self.assertEqual(configs.MODEL.MODEL_NAME,
                         configs_2.MODEL.MODEL_NAME)

        # test merging configs from other configs with invalid keys
        configs = get_dummy_config()

        configs_2 = ConfigNode()
        configs_2.DEVICE = ConfigNode()
        # invalid key that configs dose not have
        configs_2.DEVICE.DISABLE_CUDA = True
        configs_2.DEVICE.ENABLE_CUDA = False

        with self.assertRaises(KeyError):
            configs.merge_from_other_config(configs_2)

        # test merging configs from other configs with different types
        # that could be converted to match the original types
        configs = get_dummy_config()
        configs_2 = ConfigNode()
        configs_2.DEVICE = ConfigNode()
        # original is [0, 1]
        configs_2.DEVICE.DEVICE_IDS = (1, )
        configs.merge_from_other_config(configs_2)
        self.assertTrue(isinstance(configs.DEVICE.DEVICE_IDS, list))
        self.assertEqual(configs.DEVICE.DEVICE_IDS, [1, ])

        # test merging configs from other configs with different types
        # that can NOT be converted to match the original types
        configs = get_dummy_config()
        configs_2 = ConfigNode()
        configs_2.DEVICE = ConfigNode()
        # original is [0, 1]
        configs_2.DEVICE.DEVICE_IDS = 1
        with self.assertRaises(ValueError):
            configs.merge_from_other_config(configs_2)

    def test_merge_configs_from_file(self):
        """Test merging other configs from file."""
        with tempfile.NamedTemporaryFile(mode='wt') as config_file:
            configs = get_dummy_config()
            config_file.write(configs.dump())
            config_file.flush()

            configs.MODEL.MODEL_NAME = 'SVM'
            self.assertNotEqual(configs.MODEL.MODEL_NAME, 'ANN')
            self.assertEqual(configs.MODEL.MODEL_NAME, 'SVM')

            configs.merge_from_file(config_file.name)
            self.assertNotEqual(configs.MODEL.MODEL_NAME, 'SVM')
            self.assertEqual(configs.MODEL.MODEL_NAME, 'ANN')

    def test_merge_configs_from_list(self):
        """Test merging other configs from list."""
        configs = get_dummy_config()
        self.assertTrue(configs.DEVICE.ENABLE_CUDA)
        self.assertEqual(configs.DEVICE.DEVICE_IDS, [0, 1])
        self.assertEqual(configs.MODEL.MODEL_NAME, 'ANN')

        option_list = ['DEVICE.ENABLE_CUDA', False,
                       'DEVICE.DEVICE_IDS', (1, 2, 3),
                       'MODEL.MODEL_NAME', 'SVM']
        configs.merge_from_list(option_list)
        self.assertFalse(configs.DEVICE.ENABLE_CUDA)
        self.assertFalse(isinstance(configs.DEVICE.DEVICE_IDS, tuple))
        self.assertEqual(configs.DEVICE.DEVICE_IDS, [1, 2, 3])
        self.assertEqual(configs.MODEL.MODEL_NAME, 'SVM')

    def test_merge_deprecated_keys_from_file(self):
        """Test merging deprecated keys from file, which will have no effect."""
        configs = get_dummy_config()
        with self.assertRaises(AttributeError):
            _ = configs.DEPRECATED_KEY1
        with self.assertRaises(AttributeError):
            _ = configs.LOCAL.DEPRECATED_KEY2

        with tempfile.NamedTemporaryFile('wt') as configs_file:
            configs_2 = configs.clone()
            configs_2.DEPRECATED_KEY1 = 1
            configs_2.LOCAL.DEPRECATED_KEY2 = 2
            configs_file.write(configs_2.dump())
            configs_file.flush()

            configs.merge_from_file(configs_file.name)
            with self.assertRaises(AttributeError):
                _ = configs.DEPRECATED_KEY1

            with self.assertRaises(AttributeError):
                _ = configs.LOCAL.DEPRECATED_KEY2

    def test_merge_deprecated_keys_from_list(self):
        """Test merging deprecated keys from list."""
        # You should see logger messages like:
        #   "Deprecated config key (ignoring): MODEL.DILATION"
        configs = get_dummy_config()
        option_list = ['DEPRECATED_KEY1', 'foobar',
                       'LOCAL.DEPRECATED_KEY2', 0]

        with self.assertRaises(AttributeError):
            _ = configs.DEPRECATED_KEY1
        with self.assertRaises(AttributeError):
            _ = configs.LOCAL.DEPRECATED_KEY2

        configs.merge_from_list(option_list)
        with self.assertRaises(AttributeError):
            _ = configs.DEPRECATED_KEY1

        with self.assertRaises(AttributeError):
            _ = configs.LOCAL.DEPRECATED_KEY2

    def test_nonexist_key_from_list(self):
        """Test merging nonexist keys from list."""
        configs = get_dummy_config()
        option_list = ['NONEXIST_KEY1', 'foobar',
                       'LOCAL.NONEXIST_KEY2', 0]
        with self.assertRaises(AssertionError):
            configs.merge_from_list(option_list)

    def test_load_configs_with_invalid_type(self):
        """Test loading configs with value of invalid type."""
        # FOO.BAR and FOO.FOOBAR will have type None
        configs_string = 'FOO:\n BAR:\n FOOBAR: \n'
        configs = ConfigNode.load_config(configs_string)
        self.assertIsNone(configs.FOO.BAR)
        self.assertIsNone(configs.FOO.FOOBAR)

        # dict is invalid type for ConfigNode
        # however, dictionary object written in Python file work fine
        configs_dict = {'FOO.BAR.FOOBAR': 'foobar'}
        with self.assertRaises(AssertionError):
            _ = ConfigNode.load_config(configs_dict)

    def test_merge_renamed_keys_from_list(self):
        """Test merging renamed keys from list, which will raise KeyError."""
        configs = get_dummy_config()
        option_list = ['EXAMPLE.OLD.KEY', 'foobar']
        with self.assertRaises(AttributeError):
            _ = configs.EXAMPLE.OLD.KEY

        with self.assertRaises(KeyError):
            configs.merge_from_list(option_list)

    def test_renamed_key_from_file(self):
        """Test merging renamed keys from file, which will raise KeyError."""
        configs = get_dummy_config()
        with self.assertRaises(AttributeError):
            _ = configs.EXAMPLE.RENAMED.KEY

        configs_2 = configs.clone()
        configs_2.EXAMPLE = ConfigNode()
        configs_2.EXAMPLE.RENAMED = ConfigNode()
        configs_2.EXAMPLE.RENAMED.KEY = 'foobar'

        with tempfile.NamedTemporaryFile('wt') as configs_file:
            configs_file.write(configs_2.dump())
            configs_file.flush()
            with self.assertRaises(KeyError):
                configs.merge_from_file(configs_file.name)

    def test_load_configs_with_file(self):
        """Test load configs from file."""
        configs = get_dummy_config()
        with tempfile.NamedTemporaryFile('wt') as configs_file:
            configs_file.write(configs.dump())
            configs_file.flush()

            with open(configs_file.name, 'rt') as file_readed:
                configs_readed = ConfigNode.load_config(file_readed)
                self.assertEqual(configs_readed, configs)


if __name__ == "__main__":
    unittest.main()
