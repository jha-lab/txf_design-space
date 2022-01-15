from datasets import load_dataset, load_metric


GLUE_TASKS = ['cc_news', 'bookcorpus', 'wikipedia', 'openwebtext', 'c4']


def main():
    for task in GLUE_TASKS:

        if task in ['cc_news', 'bookcorpus', 'openwebtext']:
            load_dataset(task, cache_dir='../')
        elif task == 'wikipedia':
            load_dataset('wikipedia', '20200501.en', cache_dir='../')
        else:
            load_dataset('c4', 'en', cache_dir='../')

if __name__ == '__main__':
    main()


