# Graph library generating all possible graphs
# within the design space

# Author : Shikhar Tuli

import yaml
from itertools import combinations_with_replacement
from tqdm.contrib.itertools import product
import graph_util


class GraphLib(object):
    """Graph Library containing all possible graphs within the design space
    
    Attributes:
        datasets (list[str]): list of all dataset being considered for training every graph
        design_space (dict): dictionary of design space hyper-parameters
        hashes_computed (bool): is True is the library generated was checked for isomorphisms 
            in computed graphs. False otherwise, and all Graph objects have their hashes as empty strings
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
            try:
                config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                print(exc)
            self.datasets = config.get('datasets')
            self.design_space = config.get('architecture')

        # List of all possible operations in the computation graph
        self.ops_list = ['input'] 
        for hidden in self.design_space['hidden_size']:
            for sim in self.design_space['similarity_metric']:
                op = f'a_h{hidden}_s-{sim}'
                if op not in self.ops_list: self.ops_list.append(op) 
        for ff in self.design_space['feed-forward_hidden']:
            op = f'f{ff}'
            if op not in self.ops_list: self.ops_list.append(op) 
        self.ops_list.append('add_norm')
        self.ops_list.append('output')

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

    def build_library(self, check_isomorphism=False):
        """Build the GraphLib library
        
        Args:
            check_isomorphism (bool, optional): if True, isomorphism is checked 
                for every graph. Default is False, to save compute time.
        
        Raises:
            AssertionError: if two graphs are found with the same hash. Only if check_isomorphism is True
        """
        print('Creating Graph library')
        for layers in self.design_space['encoder_layers']:
            possible_a = list(combinations_with_replacement(self.design_space['attention_heads'], layers))
            possible_h = list(combinations_with_replacement(self.design_space['hidden_size'], layers))
            possible_s = list(combinations_with_replacement(self.design_space['similarity_metric'], layers))
            possible_f = list(combinations_with_replacement(self.design_space['feed-forward_hidden'], layers))
            for a, h, s, f in product(possible_a, possible_h, possible_s, possible_f, \
                    desc=f'Generating transformers with {layers} encoder layers'):
                model_dict = {'l': layers, 'a': list(a), 'h': list(h), 's': list(s), 'f': list(f)}
                new_graph = Graph(model_dict, self.datasets, self.ops_list, do_hash=check_isomorphism)
                if check_isomorphism:
                    assert new_graph.hash not in [graph.hash for graph in self.library], \
                        'Two graphs found with same hash! Check if they are isomorphic:\n' \
                        + f'Graph-1: {new_graph.model_dict}\n' \
                        + f'Graph-2: {graph.model_dict for graph in self.library if graph.hash == new_graph.hash}'
                self.library.append(new_graph)

        print(f'Graph library created! \n{len(self.library)} graphs within the design space.')

        if check_isomorphism: 
            self.hashes_computed = True
        else:
            self.hashes_computed = False


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
    def __init__(self, model_dict: dict, datasets: list, ops_list: list, do_hash: bool, hash_algo='md5'):
        """Init a Graph instance from model_dict
        
        Args:
            model_dict (dict): dictionary with the hyper-parameters for current 
                model graph
            datasets (list): list of datasets to keep accuracy for the graph
            ops_list (list): list of all possible operation blocks in 
                the computational graph
            do_hash (bool): if True, hash is computed, else, is None.
            hash_algo (str, optional): hash algorithm to use among ["md5", "sha256", "sha512"]
        """
        self.model_dict = model_dict

        # Generate model graph
        self.graph = graph_util.model_dict_to_graph(model_dict, ops_list)

        # Generate hash to check isomorphic graphs
        if do_hash: 
            self.hash = graph_util.hash_graph(*self.graph, algo=hash_algo)
        else:
            self.hash = None

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
