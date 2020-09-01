"""Average meters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict


class AverageMeter():
    """Average meter."""

    def __init__(self, name):
        self.name = name
        self.current_value = 0.0
        self.average_value = 0.0
        self.accumulated_value = 0.0
        self.accumulate_count = 0
        self.recorded_values = []

    def reset_values(self):
        """Reset values."""
        self.current_value = 0.0
        self.average_value = 0.0
        self.accumulated_value = 0.0
        self.accumulate_count = 0
        self.recorded_values = []

    def accumulate(self, current_value: float, increment=1):
        """Accumulate `increment` number of `current_value`."""
        self.current_value = current_value
        self.accumulated_value += current_value * increment
        self.accumulate_count += increment
        self.average_value = self.accumulated_value / self.accumulate_count
        for _ in range(increment):
            self.recorded_values.append(current_value)


class AverageMeters():
    """Average meters."""

    def __init__(self, names):
        self.meters = OrderedDict()
        self.names = names
        for name in names:
            self.meters[name] = AverageMeter(name)

    def reset_values(self):
        """Reset values of all meters."""
        for name in self.names:
            self.meters[name].reset_values()

    def accumulate(self, current_meters: dict, increment=1):
        """Accumulate values of all meters from dict."""
        assert isinstance(current_meters, dict)
        assert all(name in current_meters.keys() for name in self.names)
        for name in self.names:
            self.meters[name].accumulate(current_meters[name], increment)

    @property
    def average_values(self):
        """Return a dict with average values of all meters."""
        average_values = OrderedDict()
        for name in self.names:
            average_values[name] = self.meters[name].average_value

        return average_values
