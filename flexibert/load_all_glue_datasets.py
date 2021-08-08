

from datasets import load_dataset, load_metric
import argparse


GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']


def main():
    
    for task in GLUE_TASKS:

        load_dataset("glue", task)
        load_metric("glue", task)


if __name__ == '__main__':
    main()




