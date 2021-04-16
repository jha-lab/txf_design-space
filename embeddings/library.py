# Graph library generating all possible graphs
# within the design space

# Author : Shikhar Tuli

import yaml


class GraphLib(object):
    """
    Graph Library containing all possible graphs within the design space
    """
    def __init__(self, design_space: str):
        """
        Init GraphLib instance with design_space

        Args:
            design_space: yaml file containing range of hyper-parameters in
            the design space
        """
        raise NotImplementedError


class Graph(GraphLib):
    """
    Graph class to represent a computational graph in the design space
    """
    def __init__(self, model_dict: dict):
        """
        Init a Graph instance from model_dict

        Args:
            model_dict: dictionary with the hyper-parameters for current 
            model graph
        """
        super(Graph, self).__init__()

        raise NotImplementedError