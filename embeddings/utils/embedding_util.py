# Utility methods for generating embeddings

# Author : Shikhar Tuli

import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from scipy.stats import zscore


def generate_embeddings(dissimilarity_matrix, embedding_size: int, n_init=4, max_iter=1000, n_jobs=8):
    """Generate embeddings using Multi-Dimensional Scaling (SMACOF algorithm)
    
    Args:
        dissimilarity_matrix (np.ndarry): input dissimilarity_matrix
        embedding_size (int): size of the embedding
        n_init (int, optional): number of times the SMACOF algorithm will be run with 
        	different initializations. The final results will be the best output of 
        	the runs, determined by the run with the smallest final stres
        max_iter (int, optional): maximum number of iterations of the SMACOF algorithm 
        	for a single run.
        n_jobs (int, optional): number of parrallel jobs for joblib
    
    Returns:
        embeddings (np.ndarray): ndarray of embeddings of shape (len(dissimilarity_matrix), embedding_size)
    """
    #  Instantiate embedding function
    embedding_func = MDS(n_components=embedding_size, n_init=n_init, max_iter=max_iter, 
    	dissimilarity='precomputed', eps=1e-10, n_jobs=n_jobs)

    # Fit the embedding
    embeddings = embedding_func.fit_transform(dissimilarity_matrix)

    return embeddings


def generate_naive_embeddings(model_dict_list: list, design_space: dict):
    """Generate embeddings by flattening the model dictionary
    
    Args:
        model_dict_list (list): list of model dictionaries
        design_space (dict): the design space dictionary containing span
            of all the hyper-parameters, extracted from the yaml file
    
    Returns:
        embeddings (np.ndarray): ndarray of embeddings of shape (len(model_dict_list), embedding_size)
    """
    # Get maximum hyper-parameters
    max_l = max(design_space['encoder_layers'])

    # Create dictionary of similarity metrics to map to numbers
    sim_dict = {sim:idx for idx, sim in enumerate(design_space['similarity_metric'])}

    # Calculate embedding size
    embedding_size = 1 + 4 * max_l

    embeddings = np.zeros((len(model_dict_list), embedding_size))

    for i in range(len(model_dict_list)):
        embeddings[i][0] = model_dict_list[i]['l']
        for j in range(model_dict_list[i]['l']):
            for k in range(4):
                if k == 0:
                    val = model_dict_list[i]['a'][j]
                elif k == 1:
                    val = model_dict_list[i]['h'][j]
                elif k == 2:
                    val = model_dict_list[i]['f'][j]
                else:
                    val = sim_dict[model_dict_list[i]['s'][j]] 
                embeddings[i][1 + 4*k + j] = val

    return zscore(embeddings, axis=1)

def get_neighbors(embeddings):
    """Get neighbor indices for all graphs from embeddings
    
    Args:
        embeddings (np.ndarray): embeddings array from generate_embeddings() or
            generate_naive_embeddings()
    
    Returns:
        neighbor_idx (np.ndarray): neighbor indices according to the order in the
            embeddings array
    """
    # Generate distance matrix in the embeddings space
    distance_matrix = pairwise_distances(embeddings, embeddings, metric='euclidean')

    # Argmin should not return the same graph index
    np.fill_diagonal(distance_matrix, np.inf)

    # Find neighbors greedily
    neighbor_idx = np.argmin(distance_matrix, axis=0)

    return neighbor_idx