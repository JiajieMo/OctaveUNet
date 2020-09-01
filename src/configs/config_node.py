# Copyright (c) 2018-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""
Taken from [YACS -- Yet Another Configuration System](https://github.com/rbgirshick/yacs)
It is designed to be a simple configuration management system for academic and industrial research projects.
For usage and examples, see [README.md](https://github.com/rbgirshick/yacs/blob/master/README.md)

Added support for config node with None value.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import copy
import logging
from ast import literal_eval

import importlib

import yaml


# Filename extensions for loading configs from files
_YAML_EXTS = {'', '.yaml', '.yml'}
_PY_EXTS = {'.py'}

_FILE_TYPES = (io.IOBase,)

# ConfigNode can only contain a limited set of valid types
_VALID_TYPES = (tuple, list, str, int, float, bool, type(None))

LOGGER = logging.getLogger(__name__)


class ConfigNode(dict):
    """ConfigNode represents an internal node in the configuration tree.
    It's a simple dict-like container that allows for attribute-based
    access to keys.
    """

    IMMUTABLE = '__immutable__'
    DEPRECATED_KEYS = '__deprecated_keys__'
    RENAMED_KEYS = '__renamed_keys__'
    NEW_ALLOWED = '__new_allowed__'

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        """
        Args:
            init_dict (dict): the possibly-nested dictionary to initialize
                the ConfigNode.
            key_list (list[str]): a list of names which index this ConfigNode
                from the root. Currently only used for logging purposes.
            new_allowed (bool): whether adding new key is allowed when merging
                with other configs.
        """
        # Recursively convert nested dictionaries in init_dict into ConfigNode
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        init_dict = self._create_config_tree_from_dict(init_dict, key_list)

        super(ConfigNode, self).__init__(init_dict)

        # Manage if the ConfigNode is frozen or not
        self.__dict__[ConfigNode.IMMUTABLE] = False

        # Deprecated options
        # If an option is removed from the code and you don't want to break
        # existing yaml configs, you can add the full config key as a string
        # to the set below.
        self.__dict__[ConfigNode.DEPRECATED_KEYS] = set()

        # Renamed options
        # If you rename a config option, record the mapping from the old name
        # to the new name in the dictionary below. Optionally, if the type also
        # changed, you can make the value a tuple that specifies first the
        # renamed key and then instructions for how to edit the config file.
        self.__dict__[ConfigNode.RENAMED_KEYS] = {
            # 'EXAMPLE.OLD.KEY': 'EXAMPLE.NEW.KEY',  # Dummy example to follow
            # 'EXAMPLE.OLD.KEY': (                   # A more complex example
            # to follow
            #     'EXAMPLE.NEW.KEY',
            #     'Also convert to a tuple, e.g., 'foo' -> ('foo',) or '
            #     + ''foo:bar' -> ('foo', 'bar')'
            # ),
        }

        # Allow new attributes after initialization
        self.__dict__[ConfigNode.NEW_ALLOWED] = new_allowed

    @classmethod
    def _create_config_tree_from_dict(cls, dic, key_list):
        """
        Create a configuration tree using the given dict.
        Any dict-like objects inside dict will be treated as a new ConfigNode.
        Args:
            dic (dict):
            key_list (list[str]): a list of names which index this ConfigNode
                from the root. Currently only used for logging purposes.
        """
        dic = copy.deepcopy(dic)
        for key, val in dic.items():
            if isinstance(val, dict):
                # Convert dict to ConfigNode
                dic[key] = cls(val, key_list=key_list + [key])

            else:
                # Check for valid leaf type or nested ConfigNode
                _assert_with_logging(
                    _validate_type(val, allow_config_node=False),
                    'Key {} with value {} is not a valid type; '
                    'valid types: {}'.format('.'.join(key_list + [key]),
                                             type(val),
                                             _VALID_TYPES))
        return dic

    def __getattr__(self, name):
        if name in self:
            return self[name]

        raise AttributeError(name)

    def __setattr__(self, name, value):
        if self.is_frozen():
            raise AttributeError('Attempted to set {} to {}, but ConfigNode '
                                 'is immutable'.format(name, value))

        _assert_with_logging(
            name not in self.__dict__,
            'Invalid attempt to modify internal ConfigNode state: {}'.format(
                name),
        )

        _assert_with_logging(
            _validate_type(value, allow_config_node=True),
            'Invalid type {} for key {}; valid types = {}'.format(
                type(value),
                name,
                _VALID_TYPES,
            ),
        )

        self[name] = value

    def __str__(self):
        def _indent(str_, num_spaces):
            if len(str_.split('\n')) == 1:
                return str_
            str_ = str_.split('\n')
            first = str_.pop(0)
            str_ = [(num_spaces * ' ') + line for line in str_]
            str_ = '\n'.join(str_)
            str_ = first + '\n' + str_
            return str_

        repr_ = ''
        str_ = []
        for key, val in sorted(self.items()):
            seperator = '\n' if isinstance(val, ConfigNode) else ' '
            attr_str = '{}:{}{}'.format(str(key), seperator, str(val))
            attr_str = _indent(attr_str, 2)
            str_.append(attr_str)
        repr_ += '\n'.join(str_)
        return repr_

    def __repr__(self):
        repr_ = '{}({})'.format(self.__class__.__name__,
                                super(ConfigNode, self).__repr__())
        return repr_

    def dump(self, **kwargs):
        """Dump to a string."""

        def convert_to_dict(config_node, key_list):
            if not isinstance(config_node, ConfigNode):
                _assert_with_logging(
                    _validate_type(config_node, allow_config_node=False),
                    'Key {} with value {} is not a valid type; '
                    'valid types: {}'.format(
                        '.'.join(key_list),
                        type(config_node),
                        _VALID_TYPES
                    ),
                )
                return config_node

            else:
                config_dict = dict(config_node)
                for key, val in config_dict.items():
                    config_dict[key] = convert_to_dict(val, key_list + [key])
                return config_dict

        self_as_dict = convert_to_dict(self, [])
        return yaml.safe_dump(self_as_dict, **kwargs)

    def dump_to_file(self, file_path):
        """Dump to file."""

        def convert_to_dict(config_node, key_list):
            if not isinstance(config_node, ConfigNode):
                _assert_with_logging(
                    _validate_type(config_node, allow_config_node=False),
                    'Key {} with value {} is not a valid type; '
                    'valid types: {}'.format(
                        '.'.join(key_list),
                        type(config_node),
                        _VALID_TYPES
                    ),
                )
                return config_node

            else:
                config_dict = dict(config_node)
                for key, val in config_dict.items():
                    config_dict[key] = convert_to_dict(val, key_list + [key])
                return config_dict

        self_as_dict = convert_to_dict(self, [])

        with open(file_path, 'w') as outfile:
            yaml.dump(self_as_dict, outfile, default_flow_style=False)

    def merge_from_file(self, config_filename):
        """Load a yaml config file and merge it this ConfigNode."""
        with open(config_filename, 'r') as config_file:
            config = self.load_config(config_file)
        self.merge_from_other_config(config)

    def merge_from_other_config(self, config_other):
        """Merge `config_other` into this ConfigNode."""
        _merge_a_into_b(config_other, self, self, [])

    def merge_from_list(self, config_list):
        """Merge config (keys, values) in a list (e.g., from command line) into
        this ConfigNode. For example, `config_list = ['FOO.BAR', 0.5]`.
        """
        _assert_with_logging(
            len(config_list) % 2 == 0,
            'Override list has odd length: {}; it must be a list of pairs'.format(
                config_list
            ),
        )
        root = self
        for full_key, val in zip(config_list[0::2], config_list[1::2]):
            if root.key_is_deprecated(full_key):
                continue
            if root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            key_list = full_key.split('.')
            dict_ = self
            for subkey in key_list[:-1]:
                _assert_with_logging(
                    subkey in dict_, 'Non-existent key: {}'.format(full_key)
                )
                dict_ = dict_[subkey]
            subkey = key_list[-1]
            _assert_with_logging(
                subkey in dict_, 'Non-existent key: {}'.format(full_key))
            value = self.decode_config_value(val)
            value = _check_and_coerce_config_value_type(value, dict_[subkey],
                                                        full_key)
            dict_[subkey] = value

    def freeze(self):
        """Make this ConfigNode and all of its children immutable."""
        self.set_immutable(True)

    def defrost(self):
        """Make this ConfigNode and all of its children mutable."""
        self.set_immutable(False)

    def is_frozen(self):
        """Return mutability."""
        return self.__dict__[ConfigNode.IMMUTABLE]

    def set_immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested ConfigNode.
        """
        self.__dict__[ConfigNode.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for val in self.__dict__.values():
            if isinstance(val, ConfigNode):
                val.set_immutable(is_immutable)
        for val in self.values():
            if isinstance(val, ConfigNode):
                val.set_immutable(is_immutable)

    def clone(self):
        """Recursively copy this ConfigNode."""
        return copy.deepcopy(self)

    def register_deprecated_key(self, key):
        """Register key (e.g. `FOO.BAR`) a deprecated option. When merging deprecated
        keys a warning is generated and the key is ignored.
        """
        _assert_with_logging(
            key not in self.__dict__[ConfigNode.DEPRECATED_KEYS],
            'key {} is already registered as a deprecated key'.format(key),
        )
        self.__dict__[ConfigNode.DEPRECATED_KEYS].add(key)

    def register_renamed_key(self, old_name, new_name, message=None):
        """Register a key as having been renamed from `old_name` to `new_name`.
        When merging a renamed key, an exception is thrown alerting to user to
        the fact that the key has been renamed.
        """
        _assert_with_logging(
            old_name not in self.__dict__[ConfigNode.RENAMED_KEYS],
            'key {} is already registered as a renamed config key'.format(
                old_name),
        )
        value = new_name
        if message:
            value = (new_name, message)
        self.__dict__[ConfigNode.RENAMED_KEYS][old_name] = value

    def key_is_deprecated(self, full_key):
        """Test if a key is deprecated."""
        if full_key in self.__dict__[ConfigNode.DEPRECATED_KEYS]:
            LOGGER.warning(
                'Deprecated config key (ignoring): %s', full_key)
            return True
        return False

    def key_is_renamed(self, full_key):
        """Test if a key is renamed."""
        return full_key in self.__dict__[ConfigNode.RENAMED_KEYS]

    def raise_key_rename_error(self, full_key):
        """Raise key rename error."""
        new_key = self.__dict__[ConfigNode.RENAMED_KEYS][full_key]
        if isinstance(new_key, tuple):
            msg = ' Note: ' + new_key[1]
            new_key = new_key[0]
        else:
            msg = ''
        raise KeyError('Key {} was renamed to {}; please update your '
                       'config.{}'.format(full_key, new_key, msg))

    def is_new_allowed(self):
        """Return whether or not allow new key."""
        return self.__dict__[ConfigNode.NEW_ALLOWED]

    @classmethod
    def load_config(cls, config_file_obj_or_str):
        """
        Load a config.
        Args:
            config_file_obj_or_str (str or file):
                Supports loading from:
                - A file object backed by a YAML file
                - A file object backed by a Python source file that exports an
                    attribute 'config' that is either a dict or a ConfigNode
                - A string that can be parsed as valid YAML
        """
        _assert_with_logging(
            isinstance(config_file_obj_or_str, _FILE_TYPES + (str,)),
            'Expected first argument to be of type {} or {}, but got {}'.format(
                _FILE_TYPES, str, type(config_file_obj_or_str)
            ),
        )

        if isinstance(config_file_obj_or_str, str):
            return cls._load_config_from_yaml_str(config_file_obj_or_str)

        elif isinstance(config_file_obj_or_str, _FILE_TYPES):
            return cls._load_config_from_file(config_file_obj_or_str)

        else:
            raise NotImplementedError('Impossible to reach here (unless '
                                      'there are bugs)')

    @classmethod
    def _load_config_from_file(cls, file_obj):
        """Load a config from a YAML file or a Python source file."""
        _, file_extension = os.path.splitext(file_obj.name)

        if file_extension in _YAML_EXTS:
            return cls._load_config_from_yaml_str(file_obj.read())

        if file_extension in _PY_EXTS:
            return cls._load_config_py_source(file_obj.name)

        raise Exception('Attempt to load from an unsupported file type {}; '
                        'only {} are supported'.format(
                            file_obj, _YAML_EXTS.union(_PY_EXTS)))

    @classmethod
    def _load_config_from_yaml_str(cls, str_obj):
        """Load a config from a YAML string encoding."""
        config_as_dict = yaml.safe_load(str_obj)
        return cls(config_as_dict)

    @classmethod
    def _load_config_py_source(cls, filename):
        """Load a config from a Python source file."""

        def _load_module_from_file(name, filename):
            spec = importlib.util.spec_from_file_location(name, filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        module = _load_module_from_file('config.override', filename)

        _assert_with_logging(
            hasattr(module, 'config'),
            'Python module from file {} must have "config" attribute'.format(
                filename),
        )
        valid_attr_types = (dict, ConfigNode)
        _assert_with_logging(
            isinstance(module.config, (dict, ConfigNode)),
            'Imported module "config" attribute must be of type in {} '
            'but is of type {} instead'.format(
                valid_attr_types, type(module.config)
            ),
        )
        return cls(module.config)

    @classmethod
    def decode_config_value(cls, value):
        """
        Decodes a raw config value (e.g., from a yaml config files or command
        line argument) into a Python object.
        If the value is a dict, it will be interpreted as a new ConfigNode.
        If the value is a str, it will be evaluated as literals.
        Otherwise it is returned as-is.
        """
        # Configs parsed from raw yaml will contain dictionary keys that need to be
        # converted to ConfigNode objects
        if isinstance(value, dict):
            return cls(value)
        # All remaining processing is only applied to strings
        if not isinstance(value, str):
            return value
        # Try to interpret `value` as a:
        #   string, number, tuple, list, dict, boolean, or None
        try:
            value = literal_eval(value)
        # The following two excepts allow v to pass through when it represents a
        # string.
        #
        # Longer explanation:
        # The type of v is always a string (before calling literal_eval), but
        # sometimes it *represents* a string and other times a data structure,
        # like a list. In the case that v represents a string, what we got back
        # from the yaml parser is 'foo' *without quotes* (so, not ''foo'').
        # literal_eval is ok with ''foo'', but will raise a ValueError if given
        # 'foo'. In other cases, like paths (v = 'foo/bar' and not v =
        # ''foo/bar''), literal_eval will raise a SyntaxError.
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value


def _validate_type(value, allow_config_node=False):
    return (isinstance(value, _VALID_TYPES)) or (
        allow_config_node and isinstance(value, ConfigNode)
    )


def _merge_a_into_b(config_a, config_b, root, key_list):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    _assert_with_logging(
        isinstance(config_a, ConfigNode),
        '`config_a` (of type {}) must be an instance of {}'.format(
            type(config_a), ConfigNode),
    )
    _assert_with_logging(
        isinstance(config_b, ConfigNode),
        '`config_b` (of type {}) must be an instance of {}'.format(
            type(config_b), ConfigNode),
    )

    for key, val in config_a.items():
        full_key = '.'.join(key_list + [key])

        val_ = copy.deepcopy(val)
        val_ = config_b.decode_config_value(val_)

        if key in config_b:
            val_ = _check_and_coerce_config_value_type(
                val_, config_b[key], full_key)
            # Recursively merge dicts
            if isinstance(val_, ConfigNode):
                try:
                    _merge_a_into_b(
                        val_, config_b[key], root, key_list + [key])

                except BaseException as exception:
                    raise exception
            else:
                config_b[key] = val_

        elif config_b.is_new_allowed():
            config_b[key] = val_

        else:
            if root.key_is_deprecated(full_key):
                continue
            elif root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))


def _check_and_coerce_config_value_type(replacement, original, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # default None config should be replaceable for any update
    if original_type == type(None):
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)

        return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
        'key: {}'.format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def _assert_with_logging(cond, msg):
    if not cond:
        LOGGER.debug(msg)
    assert cond, msg
