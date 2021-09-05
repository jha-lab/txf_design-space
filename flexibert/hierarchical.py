# Generate dataset for next level of hierarchy

# Author : Shikhar Tuli

import sys
import os
import yaml
import json
import numpy as np
from tqdm import tqdm
import argparse

sys.path.append('../embeddings/')
sys.path.append('../boshnas/')

from library import Graph, GraphLib
from boshnas import BOSHNAS


TOP_NUM_MODELS = 5
NUM_NEIGHBORS_FOR_INTERPOLATION = 5


def main():
    parser = argparse.ArgumentParser(
        description='Input parameters for generation of dataset library',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--design_space_file',
        metavar='', 
        type=str, 
        help='path to yaml file for the design space',
        default='../design_space/design_space_test.yaml')
    parser.add_argument('--old_dataset_file',
        metavar='',
        type=str,
        help='old dataset_file',
        default='../dataset/dataset_test_bn.json')
    parser.add_argument('--new_dataset_file',
        metavar='',
        type=str,
        help='path to store the dataset',
        default='../dataset/dataset_test_bn_2.json')
    parser.add_argument('--models_dir',
        metavar='',
        type=str,
        default='../models/')
    parser.add_argument('--dataset',
        metavar='',
        type=str,
        help='name of NLP dataset',
        default=None)
    parser.add_argument('--old_layers_per_stack',
        metavar='',
        type=int,
        help='number of layers per stack',
        default=2)
    parser.add_argument('--new_layers_per_stack',
        metavar='',
        type=int,
        help='number of layers per stack',
        default=1)
    parser.add_argument('--heterogeneous_feed_forward',
        dest='heterogeneous_feed_forward',
        action='store_true',
        help='if feed-forward stacks are heterogeneous')
    parser.add_argument('--algo',
        metavar='',
        type=str,
        help='algorithm for training embeddings',
        default='GD')
    parser.add_argument('--kernel',
        metavar='',
        type=str,
        help='kernel for graph similarity computation',
        default='GraphEditDistance')
    parser.add_argument('--embedding_size',
        metavar='',
        type=int,
        help='dimensionality of embedding',
        default=16)
    parser.add_argument('--num_neighbors',
        metavar='',
        type=int,
        help='number of neighbors to save for each graph',
        default=100)
    parser.add_argument('--n_jobs',
        metavar='',
        type=int,
        help='number of parallel jobs',
        default=8)
    parser.set_defaults(heterogeneous_feed_forward=False)

    args = parser.parse_args()

    trained_hashes = os.listdir(os.path.join(args.models_dir, 'glue'))

    graphLib = GraphLib.load_from_dataset(args.old_dataset_file)

    def is_homogenous(graphObject):
        model_dict = graphObject.model_dict
        hashed_f = [hash(str(item)) for item in model_dict['f']]
        return True if len(set(model_dict['h'])) == 1 and len(set(model_dict['n'])) == 1 and len(set(model_dict['o'])) == 1 \
            and len(set(hashed_f)) == 1 and len(set(model_dict['p'])) == 1 else False

    homogenous_models, heterogenous_models = 0, 0
    X_ds_total = np.zeros((len(trained_hashes), 16))
    y_ds_total = np.zeros((len(trained_hashes)))
    count = 0

    for model_hash in trained_hashes:
        model, _ = graphLib.get_graph(model_hash=model_hash)
        X_ds_total[count, :], y_ds_total[count] = model.embedding, \
            1 - json.load(open(f'{args.models_dir}glue/{model_hash}/all_results.json'))['glue_score']
        if is_homogenous(model):
            homogenous_models += 1
        else:
            heterogenous_models += 1
        count += 1
            
    print(f'Homogenous models: {homogenous_models}\nHeterogenous models: {heterogenous_models}')

    print(f'Best design-space performance: {1 - np.amin(y_ds_total): 0.03f}')

    best_model_hash = trained_hashes[np.argmin(y_ds_total)]
    best_model, _ = graphLib.get_graph(model_hash=best_model_hash)

    print(f'Best model hash: {best_model_hash}')
    print(f'Best model dict: {best_model.model_dict}')

    print(f'Best model is homogenous: {is_homogenous(best_model)}')

    top_models = []

    print('Top 10 performances:')
    for i in range(10):
        model_hash = trained_hashes[np.argsort(y_ds_total)[i]]
        model, _ = graphLib.get_graph(model_hash=model_hash)
        top_models.append(model)
        
        print(f'{model.model_dict}: {1 - y_ds_total[np.argsort(y_ds_total)[i]]}')

    # Build new library of interpolants
    graphLib_new = GraphLib(args.design_space_file)
    new_library = []

    for i in tqdm(range(TOP_NUM_MODELS), desc='Generating new library'):
        num_neighbors = NUM_NEIGHBORS_FOR_INTERPOLATION # // (i+1)
        for n in range(num_neighbors):
            try:
                new_library.extend(graphLib.interpolate_neighbors(top_models[i], \
                    graphLib.get_graph(model_hash=top_models[i].neighbors[n])[0], \
                    args.old_layers_per_stack, args.new_layers_per_stack, \
                    heterogeneous_feed_forward=args.heterogeneous_feed_forward))
            except:
                pass

    for i in tqdm(range(TOP_NUM_MODELS), desc='Expanding new library'):
        for j in range(i+1, TOP_NUM_MODELS):
            try:
                new_library.extend(graphLib.interpolate_neighbors(top_models[i], \
                    top_models[j], args.old_layers_per_stack, args.new_layers_1, \
                    heterogeneous_feed_forward=args.heterogeneous_feed_forward))
            except:
                continue
            
    print('Length of new library: ', len(new_library))

    hashes = []
    reduced_library = []

    # Remove duplicates
    for n in tqdm(new_library, desc='Reducing new library'):
        if n.hash in hashes: continue
        hashes.append(n.hash)
        reduced_library.append(n)
        
    print('Length of reduced library: ', len(reduced_library))

    graphLib_new.library = reduced_library

    # Build embeddings
    graphLib.build_embeddings(embedding_size=args.embedding_size, algo=args.algo,
        kernel=args.kernel, neighbors=args.num_neighbors, n_jobs=args.n_jobs)
    print()

    # Save dataset
    graphLib.save_dataset(args.new_dataset_file)


if __name__ == '__main__':
    main()




