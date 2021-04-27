# Load GLUE dataset into cache. To be run with Internet connection

# Author : Shikhar Tuli

from datasets import load_dataset, load_metric
import argparse


GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']


def main():
    parser = argparse.ArgumentParser(
        description='Input parameters for generation of dataset library',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task',
        metavar='',
        type=str,
        help=f'name of GLUE tasks to train surrogate model for',
        default='sst2')

    args = parser.parse_args()

    assert args.task in GLUE_TASKS, f'GLUE task should be in: {GLUE_TASKS}'

    load_dataset("glue", args.task)
    load_metric("glue", args.task)


if __name__ == '__main__':
    main()




