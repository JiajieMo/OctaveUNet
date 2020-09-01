"""Main program."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from src.agents.base_agents.paths_handler import PathsHandler
from src.agents.retina.retinal_vessel import RetinalVesselSegmentation


# this is the main logger, do not use `name=__name__`
LOGGER = logging.getLogger()


def setup_logging(log_dir, console_log_level='INFO'):
    """Setup logging."""
    log_formatter = logging.Formatter(
        '[%(levelname)-7s] %(asctime)s - %(name)-15s: %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )

    # base logging level
    LOGGER.setLevel('DEBUG')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(log_formatter)
    LOGGER.addHandler(console_handler)

    info_file_path = os.path.join(log_dir, 'info.log')
    info_file_handler = logging.FileHandler(info_file_path)
    info_file_handler.setLevel('INFO')
    info_file_handler.setFormatter(log_formatter)
    LOGGER.addHandler(info_file_handler)

    debug_file_path = os.path.join(log_dir, 'debug.log')
    debug_file_handler = logging.FileHandler(debug_file_path)
    debug_file_handler.setLevel('DEBUG')
    debug_file_handler.setFormatter(log_formatter)
    LOGGER.addHandler(debug_file_handler)


def main():
    """Main program."""

    # get path for main logger
    paths_handler = PathsHandler()
    paths_handler.handle_paths()

    # setup logging
    setup_logging(log_dir=paths_handler.paths['log_dir'],
                  console_log_level='INFO')

    # add get_agents when multiple agents are available
    agent = RetinalVesselSegmentation()

    agent.run()


if __name__ == '__main__':
    main()
