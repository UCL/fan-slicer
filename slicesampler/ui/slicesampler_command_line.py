# coding=utf-8

"""Command line processing"""


import argparse
from slicesampler import __version__
from slicesampler.ui.slicesampler_demo import run_demo


def main(args=None):
    """Entry point for slicesampler application"""

    parser = argparse.ArgumentParser(description='slicesampler')

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='slicesampler version ' + friendly_version_string)

    args = parser.parse_args(args)

    run_demo()
