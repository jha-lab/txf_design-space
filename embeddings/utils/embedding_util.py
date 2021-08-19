# Utility methods for generating embeddings

# Author : Shikhar Tuli

import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from scipy.stats import zscore
from tqdm import tqdm

from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def generate_mds_embeddings(dissimilarity_matrix, embedding_size: int, n_init=4, max_iter=1000, n_jobs=8):
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


def generate_grad_embeddings(dissimilarity_matrix, embedding_size: int, epochs: int = 10, 
		batch_size: int = 2048, n_jobs=8, silent: bool = False):
    """Generate embeddings using Gradient Descent on GPU
    
    Args:
        dissimilarity_matrix (np.ndarray): input dissimilarity matrix
        embedding_size (int): size of the embedding
        epochs (int, optional): number of epochs
        batch_size (int, optional): batch size for the number of pairs to consider
        n_jobs (int, optional): number of parallel jobs	
        silent (bool, optional): whether to suppress output
    
    Returns:
        embeddings (np.ndarray): ndarray of embeddings of shape (len(dissimilarity_matrix), embedding_size)
    """
    torch.manual_seed(0)
    
    # Create the model which learns graph embeddings
    class GraphEmbeddingModel(nn.Module):
        def __init__(self, num_graphs, embedding_size):
            super().__init__()
            self.embeddings = nn.Embedding(num_graphs, embedding_size)

        def forward(self, model_idx_pairs):
            embeddings_first = self.embeddings(model_idx_pairs[:, 0])
            embeddings_second = self.embeddings(model_idx_pairs[:, 1])
            distances = F.pairwise_distance(embeddings_first, embeddings_second)
            return distances

    # Create distance dataset
    class DistanceDataset(Dataset):
        def __init__(self, distance_matrix):
            super().__init__()
            self.num_pairs = int(0.5 * len(distance_matrix) * (len(distance_matrix) - 1))
            self.graph_pairs = torch.zeros([self.num_pairs, 2], dtype=torch.int64)
            self.labels = torch.zeros(self.num_pairs)

            count = 0
            for i, j in tqdm(list(combinations(range(len(distance_matrix)), 2)), desc='Generating dataset'):
                    self.graph_pairs[count, :] = torch.Tensor([i, j])
                    self.labels[count] = distance_matrix[i, j]
                    count += 1 

        def __getitem__(self, pair_idx):
            return {'graph_pairs': self.graph_pairs[pair_idx, :], 
                    'labels': self.labels[pair_idx]}

        def __len__(self):
            return self.num_pairs

    # Create dataloader
    train_loader = DataLoader(DistanceDataset(dissimilarity_matrix), 
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=n_jobs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model
    model = GraphEmbeddingModel(len(dissimilarity_matrix), embedding_size)
    model.to(device)
    model.train()

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # Instantiate the scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10)

    losses = []
    report_interval = 100

    # Run training
    for epoch in tqdm(range(epochs), desc='Training embeddings'):
        if not silent: print(f'Epoch: {epoch}')

        for i, data_batch in enumerate(train_loader):
            graph_pairs = data_batch['graph_pairs'].to(device, non_blocking=True)
            labels = data_batch['labels'].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(graph_pairs)
            loss = F.mse_loss(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i/len(train_loader))

            losses.append(loss.item())

            if not silent and i > 0 and i % report_interval == 0:
                print(f'\t[{i: 6d}/{len(train_loader): 6d}] Loss: {np.mean(losses[-report_interval:]): 0.6f}')

    embeddings = np.zeros((len(dissimilarity_matrix), embedding_size))

    model.to("cpu")
    with torch.no_grad():
        for i in range(len(dissimilarity_matrix)):
            embeddings[i, :] = model.embeddings.weight[i].numpy()

    return embeddings


def generate_naive_embeddings(model_dict_list: list, design_space: dict, compute_zscore=True):
    """Generate embeddings by flattening the model dictionary
    
    Args:
        model_dict_list (list): list of model dictionaries
        design_space (dict): the design space dictionary containing span
            of all the hyper-parameters, extracted from the yaml file
        compute_zscore (bool, optional): to compute the zscore or not
    
    Returns:
        embeddings (np.ndarray): ndarray of embeddings of shape (len(model_dict_list), embedding_size)
    """
    # Get maximum hyper-parameters
    max_l = max(design_space['encoder_layers'])

    # Create dictionary of operation types to map to numbers
    op_dict = {op:idx for idx, op in enumerate(design_space['operation_types'])}

    # Create a dictionary of operation parameters to map to numbers
    par_dict = {}
    count = 1
    for op in design_space['operation_types']:
        for par in design_space['operation_parameters'][op]:
            par_dict[par] = count
            count += 1

    # Calculate embedding size
    embedding_size = 1 + 5 * max_l

    embeddings = np.zeros((len(model_dict_list), embedding_size))

    for i in range(len(model_dict_list)):
        embeddings[i][0] = model_dict_list[i]['l']
        for j in range(model_dict_list[i]['l']):
            for k in range(5):
                if k == 0:
                    val = model_dict_list[i]['h'][j]
                elif k == 1:
                    val = model_dict_list[i]['n'][j]
                elif k == 2:
                    val = op_dict[model_dict_list[i]['o'][j]]
                elif k == 3:
                    val = par_dict[model_dict_list[i]['p'][j]]
                else:
                    val = 1
                    for f in model_dict_list[i]['f'][j]:
                        val *= f
                embeddings[i][1 + k + 5*j] = val

    if compute_zscore:
        return zscore(embeddings, axis=1)
    else:
        return embeddings

def get_neighbors(embeddings, method: str, graph_list: list, neighbors: int):
    """Get neighbor indices for all graphs from embeddings
    
    Args:
        embeddings (np.ndarray): embeddings array from generate_embeddings() or
            generate_naive_embeddings()
        method (str): one in 'distance' or 'biased'
        graph_list (list): list of graphs in the library. Required if
            method = 'biased'
        neighbors (int): number of neighbors
    
    Returns:
        neighbor_idx (np.ndarray): neighbor indices according to the order in the
            embeddings array
    """
    if method == 'distance':
        # Generate distance matrix in the embeddings space
        distance_matrix = pairwise_distances(embeddings, embeddings, metric='euclidean')

        # Argmin should not return the same graph index
        np.fill_diagonal(distance_matrix, np.inf)

        # Find neighbors greedily
        neighbor_idx = np.argsort(distance_matrix, axis=1)[:, :neighbors]
    elif method == 'biased':
        # graph_list is required
        assert graph_list is not None, 'graph_list is required with method = "biased"'

        # Get biased overlap between graphs
        def _get_overlap(curr_graph, query_graph):
            overlap = 0
            for i in range(len(curr_graph[1])):
                # Basic test
                if i >= len(query_graph[1]): break

                # Check if ops are the same
                if curr_graph[1][i] == query_graph[1][i]:
                    overlap += 1
                else:
                    break

            return overlap

        neighbor_idx = np.zeros((embeddings.shape[0], neighbors))
        ids = list(range(len(graph_list)))

        for i in tqdm(range(len(graph_list)), desc='Generating biased neighbors'):
            # Sort indices based on overlap. Break ties with embedding distance
            neighbor_idx[i, :] = sorted(ids, key=lambda idx: (-1*_get_overlap(graph_list[i], graph_list[idx]), \
                float(pairwise_distances(embeddings[i, :].reshape(1, -1), embeddings[idx, :].reshape(1, -1)))))[1:neighbors+1]

    return neighbor_idx