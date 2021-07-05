# Utility functions for colored printing and logging

# Author :  Shikhar Tuli

import math

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

def human_format(num, precision=2, separator=''):
    magnitude = 0

    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    suffixes = ['', 'K', 'M', 'B', 'T']
    # Kilo, Million, Billion, Trillion

    return '{num:.{precision}f}{separator}{suffix}'.format(
        num=num, precision=precision, separator=separator, suffix=suffixes[magnitude])