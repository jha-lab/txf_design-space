# Graph library generating all possible graphs
# within the design space

# Author : Shikhar Tuli

import argparse
from os import path
from library import GraphLib, Graph
from utils import print_util as pu

def main():
    parser = argparse.ArgumentParser(
        description='Input parameters for generation of dataset library',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--design_space_file',
        metavar='', 
        type=str, 
        help='path to yaml file for the design space')
    parser.add_argument('--dataset_file',
        metavar='',
        type=str,
        help='path to store the dataset')
    parser.add_argument('--dataset',
        metavar='',
        type=str,
        help='name of NLP dataset',
        default=None)
    parser.add_argument('--kernel',
        metavar='',
        type=str,
        help='kernel for graph similarity computation',
        default='GraphEditDistance')
    parser.add_argument('--layers_per_stack',
        metavar='',
        type=int,
        help='number of layers per stack',
        default=2)
    parser.add_argument('--algo',
        metavar='',
        type=str,
        help='algorithm for training embeddings',
        default='GD')
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
    parser.add_argument('--diss_mat_file',
        metavar='',
        type=str,
        help='path to the dissimilarity matrix file',
        default=None)
    parser.add_argument('--n_jobs',
        metavar='',
        type=int,
        help='number of parallel jobs',
        default=8)

    args = parser.parse_args()

    if not path.exists(args.dataset_file):
        # Create an empty graph library with the hyper-parameter ranges
        # given in the design_space file
        graphLib = GraphLib(args.design_space_file, args.dataset)

        # Show generated library
        print(f'{pu.bcolors.OKGREEN}Generated empty library{pu.bcolors.ENDC}')
        print(graphLib)
        print()

        # Generating graph library
        graphLib.build_library(layers_per_stack=args.layers_per_stack)
        print()

        # Simple test to check isomorphisms, rather than comparing hash for every new graph
        hashes = [graph.hash for graph in graphLib.library]
        if len(hashes) == len(set(hashes)):
            print(f'{pu.bcolors.OKGREEN}No isomorphisms detected!{pu.bcolors.ENDC}')
        else:
            print(f'{pu.bcolors.WARNING}Graphs with the same hash encountered!{pu.bcolors.ENDC}')
        print()

        # Save dataset without embeddings to dataset_file
        graphLib.save_dataset(args.dataset_file)
        print()
    else:
        # Load dataset without embeddings from dataset_file
        graphLib = GraphLib.load_from_dataset(args.dataset_file)

        print(f'{pu.bcolors.OKGREEN}Dataset without embeddings loaded from:{pu.bcolors.ENDC}' \
            + f' {args.dataset_file}')
        print()

    # Build embeddings
    graphLib.build_embeddings(embedding_size=args.embedding_size, algo=args.algo,
        kernel=args.kernel, diss_mat_file=args.diss_mat_file, neighbors=args.num_neighbors, n_jobs=args.n_jobs)
    print()

    # Save dataset
    graphLib.save_dataset(args.dataset_file)


if __name__ == '__main__':
    main()




