# Graph library generating all possible graphs
# within the design space

# Author : Shikhar Tuli

import yaml
from itertools import combinations_with_replacement, product
import graph_util


class GraphLib(object):
    """Graph Library containing all possible graphs within the design space
    
    Attributes:
        datasets (list[str]): list of all dataset being considered for training every graph
        design_space (dict): dictionary of design space hyper-parameters
        library (list[Graph]): list of all possible graphs within the design space
        ops_list (list[str]): list of all possible operation blocks in the computational graph
    """
    
    def __init__(self, design_space: str):
        """Init GraphLib instance with design_space
        
        Args:
            design_space (str): path to yaml file containing range of hyper-parameters in
                the design space

        Raises:
            AssertionError: if a sanity check fails
        """
        with open(design_space) as config_file:
            config = yaml.load(config_file)
            self.datasets = config.get('datasets')
            self.design_space = config.get('design_space')

        # List of all possible operations in the computation graph
        self.ops_list = [] 
        for hidden in self.design_space['h']:
            for sim in self.design_space['s']:
                op = f'a_h{hidden}_s-{sim}'
                self.op_lists.append(op) if op not in self.ops_list
        for ff in range(self.design_space['f']):
            op = f'f{ff}'
            self.ops_list.append(op) if op not in self.ops_list
        self.ops_list.append('add_norm')

        # Check for no duplicate operations
        assert len(self.ops_list) == len(set(self.ops_list)), \
            f'Operations list contains duplicates:\n{self.ops_list}'

        # Library of all graphs
        self.library = []

    def __len__(self):
        """Computes the number of graphs in the library

        Returns:
            len (int): number of graphs in the library
        """
        return len(self.library)

    def __repr__(self):
        """Representation of the GraphLib"""
        return f'Graph Library with design space:\n{self.design_space}\nNumber of graphs: {len(self.library)}'

    def build_library(self):
        """Build the GraphLib library

        Raises:
            AssertionError: if two graphs are found with the same hash
        """
        for layers in self.design_space['l']:
            possible_a = list(combinations_with_replacement(self.design_space['a'], layers))
            possible_h = list(combinations_with_replacement(self.design_space['h'], layers))
            possible_s = list(combinations_with_replacement(self.design_space['s'], layers))
            possible_f = list(combinations_with_replacement(self.design_space['f'], layers))
            for a, h, s, f in product(possible_a, possible_h, possible_s, possible_f):
                model_dict = {'l': layers, 'a': a, 'h': h, 's': s, 'f': f}
                new_graph = Graph(model_dict, self.datasets, self.ops_list)
                assert new_graph.hash not in [graph.hash for graph in self.library], \
                    'Two graphs found with same hash! Check if they are isomorphic:\n' \
                    + f'Graph-1: {new_graph.model_dict}\n' \
                    + f'Graph-2: {graph.model_dict for graph in self.library if graph.hash == new_graph.hash}'
                self.library.append(Graph(model_dict))

        print(f'Graph library created! \n{len(self.library)} graphs within the design space.')


class Graph(object):
    """Graph class to represent a computational graph in the design space
    
    Attributes:
        accuracy (dict): dictionary of accuracies for all datasets in consideration
        embedding (np.ndarray): embedding for every graph in the design space
        graph (tuple(np.ndarray, list[str])): model graph as a tuple of adjacency matrix and 
            a list of operations
        hash (str): hash for current graph to check isomorphism
        model_dict (dict): dictionary of model hyper-parameter choices for this graph
        ops_idx (list[int]): list of operation indices
    """
    def __init__(self, model_dict: dict, datasets: list, ops_list: list, hash_algo='md5'):
        """Init a Graph instance from model_dict
        
        Args:
            model_dict (dict): dictionary with the hyper-parameters for current 
                model graph
            datasets (list): list of datasets to keep accuracy for the graph
            ops_list (list): list of all possible operation blocks in 
                the computational graph
            hash_algo (str, optional): hash algorithm to use among ["md5", "sha256", "sha512"]
        """
        self.model_dict = model_dict

        # Generate model graph
        self.graph = graph_util.model_dict_to_graph(model_dict, ops_list, algo=hash_algo)

        # Generate hash to check isomorphic graphs
        self.hash = graph_util.hash_graph(*self.graph)

        # Keep operation indices for similarity metrics
        self.ops_idx = [ops_list.index(op) for op in self.graph[1]]

        # Initialize embedding
        self.embedding = None

        # Initialize accuracies for all datasets
        self.accuracy = {dataset:None for dataset in datasets}

    def __repr__(self):
        """Representation of the Graph"""
        return f'Graph model_dict: {self.model_dict}\n' \
            + f'Accuracies: {self.accuracy}\n' \
            + f'Embedding: {self.embedding}\n' \
            + f'Hash: {self.hash}'
