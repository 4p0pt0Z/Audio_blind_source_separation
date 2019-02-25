import argparse
import math


def str2bool(v):
    """
        Robustly convert a command line argument to a boolean value
    Args:
        v (str): command line argument

    Returns:
        Boolean value of the input
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def next_power_of_2(x):
    """Find the first power of 2 greater than x (x positive)"""
    assert x >= 0
    if x == 0:
        return 1
    else:
        return 2 ** math.ceil(math.log2(x))
