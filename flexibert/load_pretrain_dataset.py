
from datasets import load_dataset, load_metric
import argparse
from datasets data_lo



GLUE_TASKS = ['cc_news','bookcorpus','wikipedia','openwebtext']


def main():

    parser = argparse.ArgumentParser(
        description='Input parameters for pretraining',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--id',
        metavar='',
        type=str,
        help='NetID',
        default='bdedhia')

    args = parser.parse_args()



    for task in GLUE_TASKS:

        if task != 'wikipedia':
            load_dataset(task,'plaintext',cache_dir='/scratch/gpfs/'+args.id+'/'+task)
            #load_metric(task)
        else:
            load_dataset('wikipedia','20200501.en',cache_dir='/scratch/gpfs/'+args.id+'/'+task)
            #load_metric('wikipedia')

if __name__ == '__main__':
    main()


