# Utility methods for managing graphs

# Author : Shikhar Tuli

import os
import sys

import numpy as np
import hashlib

from grakel import Graph, WeisfeilerLehman, NeighborhoodHash, RandomWalkLabeled
import networkx as nx
import re

from tqdm import tqdm
from itertools import combinations, combinations_with_replacement, product
from joblib import Parallel, delayed, dump, load
import shutil

import scipy
import random


SUPPORTED_KERNELS = ['WeisfeilerLehman', 'NeighborhoodHash', 'RandomWalkLabeled', 'GraphEditDistance']
PARALLELIZE = False
USE_MEMMAP = True # Only used if PARALLELIZE is True
CHUNK_COMPUTE = False
CHUNK_SIZE = 10 # Only used if CHUNK_COMPUTE is True
RANDOM_FRAC = 0.01 # Only used is CHUNK_COMPUTE is False

DISS_MAT_TEMP = '/scratch/gpfs/stuli/diss_mat_temp.npy'
IDX_SET_TEMP = '/scratch/gpfs/stuli/idx_set_temp.npy'


def model_dict_to_graph(model_dict, ops_list = None):
	"""Converts model_dict to model_graph which is a tuple of
	adjacency matrix and ops

	Args:
		model_dict: the model dictionary
		ops_list: list of all possible operations

	Returns:
		model_graph = (adjacency_matrix, ops)

	Raises:
		ValueError: if a required operation is not in ops_list
		AssertionError: if a sanity check fails
	"""
	if 'p' in model_dict.keys():
		from_model_dict_hetero = False
	else:
		from_model_dict_hetero = True

	layers = model_dict.get('l')
	operation = model_dict.get('o')
	hidden = model_dict.get('h')

	if from_model_dict_hetero:
		num_heads = [len(layer) for layer in model_dict['o']]
	else:
		num_heads = model_dict.get('n')

	feed_forward = model_dict.get('f')
	num_feed_forward = [len(feed_forward[layer]) for layer in range(layers)]
	
	if not from_model_dict_hetero:
		parameter = model_dict.get('p')

	if from_model_dict_hetero:
		assert layers == len(operation) == len(hidden) == len(num_heads) == len(feed_forward), \
			f'Input model_dict is incorrect:\n{model_dict}'
	else:
		assert layers == len(operation) == len(hidden) == len(num_heads) == len(feed_forward) == len(parameter), \
			f'Input model_dict is incorrect:\n{model_dict}'

	V = 1 # One vertex for the input
	for layer in range(layers):
		# Input goes into a few operation heads
		# Then, the output of the operation block goes to add-norm
		# This then goes to a feed-forward layer
		# Then we have the last add-norm layer after the feed-forward layer
		# These feed-forward layers are stacked multiple times
		V += num_heads[layer] + 1 + num_feed_forward[layer] + 1
	V += 1 # Finally, the output vertex

	matrix = np.zeros((V, V), dtype=int)

	# Generate operations list
	ops = ['input']
	for layer in range(layers): 
		for head in range(num_heads[layer]):
			if from_model_dict_hetero:
				op = model_dict['o'][layer][head]
			else:
				op = f'{operation[layer]}_h{hidden[layer]}_p-{parameter[layer]}'
			if ops_list is not None and op not in ops_list:
				raise ValueError(f'Operation: {op}, not in ops_list')
			else:
				ops.append(op)
		ops.append('add_norm')
		for f in range(num_feed_forward[layer]):
			op = f'f{feed_forward[layer][f]}'
			if ops_list is not None and op not in ops_list:
				raise ValueError(f'Operation: {op}, not in ops_list')
			ops.extend([op])
		ops.append('add_norm')
	ops.append('output')

	assert len(ops) == V, \
		f'Number of operations is not equal to size of the adjacency matrix:\nOperations: {ops}\nV: {V}'

	# Generate adjacency matrix
	i = 0
	while i < len(ops) - 2:
		for layer in range(layers):
			for h in range(num_heads[layer]):
				matrix[i][i + 1 + h] = 1 # Input to all operation heads
				matrix[i + 1 + h][i + 1 + num_heads[layer]] = 1 # operation head to add_norm
			matrix[i][i + 1 + num_heads[layer]] = 1 # residual to add_norm
			i += num_heads[layer] + 1 # Re-instate i to add_norm
			for f in range(num_feed_forward[layer]):
				matrix[i + f][i + f + 1] = 1 # add_norm to feed-forward or feed-forward to next
			matrix[i + num_feed_forward[layer]][i + num_feed_forward[layer] + 1] = 1 # last feed-forward to add_norm
			matrix[i][i + num_feed_forward[layer] + 1] = 1 # residual to add_norm
			i += num_feed_forward[layer] + 1 # Re-instate i to add_norm as output of current encoder layer
	matrix[-2][-1] = 1 # add_norm to output
 
	assert is_upper_tri(matrix), \
		f'Matrix generated is not an upper-triangular adjacency matrix: \n{matrix}'
	assert is_full_dag(matrix), \
		f'Matrix generated has "hanging" vertices: \n{matrix}'

	return (matrix, ops)


def is_upper_tri(matrix):
	"""Checks if the given matrix is an upper-triangular adjacency matrix
	
	Args:
		matrix: A numpy matrix

	Returns:
		True if matrix is an upper-triangular adjacency matrix
	"""
	return np.allclose(matrix, np.triu(matrix)) \
		and np.amax(matrix) == 1 \
		and np.amin(matrix) == 0


def is_full_dag(matrix):
	"""Full DAG == all vertices on a path from vert 0 to (V-1).
	i.e. no disconnected or "hanging" vertices.

	It is sufficient to check for:
	1) no rows of 0 except for row V-1 (only output vertex has no out-edges)
	2) no cols of 0 except for col 0 (only input vertex has no in-edges)

	Args:
		matrix: V x V upper-triangular adjacency matrix

	Returns:
		True if the there are no dangling vertices.
	"""
	shape = np.shape(matrix)

	rows = matrix[:shape[0]-1, :] == 0
	rows = np.all(rows, axis=1)     # Any row with all 0 will be True
	rows_bad = np.any(rows)

	cols = matrix[:, 1:] == 0
	cols = np.all(cols, axis=0)     # Any col with all 0 will be True
	cols_bad = np.any(cols)

	return (not rows_bad) and (not cols_bad)


def num_edges(matrix):
	"""Computes number of edges in adjacency matrix."""
	return np.sum(matrix)


def hash_func(str, algo='md5'):
  """Outputs has based on algorithm defined."""
  return eval(f"hashlib.{algo}(str)")


def hash_graph(matrix, ops, algo='md5', model_dict=None):
	"""Computes a graph-invariant hash of the matrix and ops pair.

	Args:
		matrix: np.ndarray square upper-triangular adjacency matrix.
		ops: list operations of length equal to both dimensions of
			matrix.
		algo: hash algorithm among ["md5", "sha256", "sha512"]
		model_dict (optional): the model dictionary; optional for backward
			compatibility with FlexiBERT

	Returns:
		Hash of the matrix and ops based on algo.
	"""
	vertices = np.shape(matrix)[0]
	in_edges = np.sum(matrix, axis=0).tolist()
	out_edges = np.sum(matrix, axis=1).tolist()

	assert len(in_edges) == len(out_edges) == len(ops)
	hashes = list(zip(out_edges, in_edges, ops))
	hashes = [hash_func(str(h).encode('utf-8'), algo).hexdigest() for h in hashes]

	# Computing this up to the diameter is probably sufficient but since the
	# operation is fast, it is okay to repeat more times.
	for _ in range(5):
		new_hashes = []
		for v in range(vertices):
			in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
			out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
			new_hashes.append(hash_func(
				(''.join(sorted(in_neighbors)) + '|' +
				''.join(sorted(out_neighbors)) + '|' +
				hashes[v]).encode('utf-8'), algo).hexdigest())
		hashes = new_hashes

	fingerprint = hash_func(str(sorted(hashes)).encode('utf-8'), algo).hexdigest()

	if model_dict is not None:
		fingerprint = hash_func(str(fingerprint + str(model_dict['h']) + str(model_dict['f'])).encode('utf-8'), algo).hexdigest()

	return fingerprint


def generate_dissimilarity_matrix(graph_list: list, kernel='WeisfeilerLehman', ops_list: list = None, n_jobs=8, approx=1):
	"""Generate the dissimilarity matrix which is N x N, for N graphs 
	in the design space
	
	Args:
		graph_list (list): list of graphs, which are 
			tuples of adjacency matrix and ops
		kernel (str, optional): the kernel to be used for computing the dissimilarity matrix. The
			default value is 'WeisfeilerLehman'
		ops_list (list, optional): list of operations supported. Required when kernel = 'GraphEditDistance'
		n_jobs (int, optional): number of parrallel jobs for joblib
		approx (int, optional): number of approximations to be implemented. Used when kernel = 'GraphEditDistance'
	
	Returns:
		dissimilarity_matrix (np.ndarray): dissimilarity matrix
	"""

	assert kernel in SUPPORTED_KERNELS, f'Kernel {kernel} is not supported yet. Use either of: {SUPPORTED_KERNELS}'
	
	if kernel in ['WeisfeilerLehman', 'NeighborhoodHash', 'RandomWalkLabeled']:
		grakel_graph_list = []
		for graph in graph_list:
			grakel_graph_list.append(Graph(graph[0], 
				node_labels={idx:op for idx, op in zip(range(len(graph[1])), graph[1])}))

		# Instantiate kernel function based on choice of distance kernel
		kernel_func = eval(f'{kernel}(n_jobs={n_jobs}, normalize=True)')

		# Generate similarity matrix based on kernel function
		similarity_matrix = kernel_func.fit_transform(grakel_graph_list)

		dissimilarity_matrix = 1 - similarity_matrix

	elif kernel == 'GraphEditDistance':
		assert ops_list is not None, '"ops_list" is required when kernel = "GraphEditDistance"'

		sorted_ops_list = ['input', 'output', 'add_norm', '']
		att_sizes = set([int(re.search('h([0-9]+)', op).group(0)[1:]) for op in ops_list if op.startswith('sa_')])
		lin_sizes = set([int(re.search('h([0-9]+)', op).group(0)[1:]) for op in ops_list if op.startswith('l_')])
		conv_sizes = set([int(re.search('h([0-9]+)', op).group(0)[1:]) for op in ops_list if op.startswith('c_')])
		kernel_sizes = set([int(re.search('p-([0-9]+)', op).group(0)[2:]) for op in ops_list if op.startswith('c_')])
		f_sizes = set([int(re.search('f([0-9]+)', f).group(0)[1:]) for f in ops_list if f.startswith('f')])

		for f in f_sizes:
			sorted_ops_list.append(f'f{f}')

		sorted_ops_list.append('')

		for l in lin_sizes:
			sorted_ops_list.append(f'l_h{l}_p-dft')
			sorted_ops_list.append(f'l_h{l}_p-dct')

		sorted_ops_list.append('')

		for a in att_sizes:
			sorted_ops_list.append(f'sa_h{a}_p-sdp')
			sorted_ops_list.append(f'sa_h{a}_p-wma')

		sorted_ops_list.append('')

		for c in conv_sizes:
			for k in sorted(kernel_sizes):
				sorted_ops_list.append(f'c_h{c}_p-{k}')

		nx_graph_list = []

		for graph in graph_list:
			nx_graph = nx.DiGraph(graph[0])
			nx.set_node_attributes(nx_graph, {i:label for i, label in enumerate(graph[1])}, 'label')
			nx_graph_list.append(nx_graph)

		dissimilarity_matrix = np.empty((len(graph_list), len(graph_list)))
		dissimilarity_matrix[:] = np.NaN

		if os.path.exists(DISS_MAT_TEMP):
			dissimilarity_matrix = np.load(open(DISS_MAT_TEMP, 'rb'))

		def node_subst_cost(node1, node2):
			if node1['label'] == node2['label']:
				return 0
			else:
				return abs(sorted_ops_list.index(node1['label']) - sorted_ops_list.index(node2['label']))
				
		def node_cost(node):
			return sorted_ops_list.index(node['label'])

		def edge_cost(edge):
			return 0.1

		def get_ged(i, j, dissimilarity_matrix, approx=approx):

			if not CHUNK_COMPUTE:
				if approx == 0:
					dissimilarity_matrix[i, j] = nx.graph_edit_distance(nx_graph_list[i], nx_graph_list[j],
																		node_subst_cost=node_subst_cost, 
																		node_del_cost=node_cost, 
																		node_ins_cost=node_cost, 
																		edge_del_cost=edge_cost,
																		edge_ins_cost=edge_cost,
																		timeout=10)
				else:
					count = 0
					approx_dist = 0
					for dist in nx.optimize_graph_edit_distance(nx_graph_list[i], nx_graph_list[j],
																node_subst_cost=node_subst_cost, 
																node_del_cost=node_cost, 
																node_ins_cost=node_cost, 
																edge_del_cost=edge_cost,
																edge_ins_cost=edge_cost):
						approx_dist = dist
						count += 1
						if count == approx: break

					dissimilarity_matrix[i, j] = approx_dist
			else:
				if approx == 0:
					for i_idx, j_idx in product(list(range(i, i + CHUNK_SIZE)), list(range(j, j + CHUNK_SIZE))):
						dissimilarity_matrix[i_idx, j_idx] = nx.graph_edit_distance(nx_graph_list[i_idx], nx_graph_list[j_idx],
																			node_subst_cost=node_subst_cost, 
																			node_del_cost=node_cost, 
																			node_ins_cost=node_cost, 
																			edge_del_cost=edge_cost,
																			edge_ins_cost=edge_cost,
																			timeout=10)
				else:
					for i_idx, j_idx in product(list(range(i, i + CHUNK_SIZE)), list(range(j, j + CHUNK_SIZE))):
						if i_idx >= dissimilarity_matrix.shape[0] or j_idx >= dissimilarity_matrix.shape[1]: continue
						count = 0
						approx_dist = 0
						for dist in nx.optimize_graph_edit_distance(nx_graph_list[i_idx], nx_graph_list[j_idx],
																	node_subst_cost=node_subst_cost, 
																	node_del_cost=node_cost, 
																	node_ins_cost=node_cost, 
																	edge_del_cost=edge_cost,
																	edge_ins_cost=edge_cost):
							approx_dist = dist
							count += 1
							if count == approx: break

						dissimilarity_matrix[i_idx, j_idx] = approx_dist

		if not PARALLELIZE:
			if not CHUNK_COMPUTE:
				idx_set = list(combinations(range(len(graph_list)), 2))
				if RANDOM_FRAC:
					idx_set = random.sample(idx_set, int(len(idx_set)*RANDOM_FRAC))
					# Adding top-right point to aid interpolation
					idx_set.append((0, len(graph_list) - 1))
					idx_set = list(set(idx_set))

					if not os.path.exists(IDX_SET_TEMP):
						np.save(open(IDX_SET_TEMP, 'wb+'), idx_set)
					else:
						idx_set = np.load(open(IDX_SET_TEMP, 'rb'))
						print(f'Loaded idx_set from "{IDX_SET_TEMP}"')

				for i, j in tqdm(idx_set, desc='Generating dissimilarity matrix'):
					count = 0
					if np.isnan(dissimilarity_matrix[i, j]):
						get_ged(i, j, dissimilarity_matrix)
						count += 1
						if count % 100 == 0:
							np.save(open(DISS_MAT_TEMP, 'wb+'), dissimilarity_matrix)
							count = 0
			else:
				for i, j in tqdm(list(combinations(range(0, len(graph_list), CHUNK_SIZE), 2)), \
					desc='Generating dissimilarity matrix'):
					get_ged(i, j, dissimilarity_matrix)
		else:
			# It is found empirically that serial operations are faster
			if USE_MEMMAP:
				folder = './temp'
				try:
					os.mkdir(folder)
				except FileExistsError:
					pass
				memmap_file = os.path.join(folder, 'diss_mat_memmap')
				if os.path.exists(memmap_file): os.remove(memmap_file)
				dissimilarity_matrix = np.memmap(memmap_file, shape=(len(graph_list), len(graph_list)), mode='w+')

			if not CHUNK_COMPUTE:
				idx_set = list(combinations(range(len(graph_list)), 2))
				if RANDOM_FRAC:
					idx_set = random.sample(idx_set, int(len(idx_set)*RANDOM_FRAC))
					# Adding top-right point to aid interpolation
					idx_set.append((0, len(graph_list) - 1))
					idx_set = list(set(idx_set))
				if not USE_MEMMAP:
					with Parallel(n_jobs=n_jobs, prefer='threads', require='sharedmem') as parallel:
						parallel(delayed(get_ged)(i, j, dissimilarity_matrix) \
							for i, j in tqdm(idx_set, \
								desc='Generating dissimilarity matrix'))
				else:
					with Parallel(n_jobs=n_jobs, prefer='threads') as parallel:
						parallel(delayed(get_ged)(i, j, dissimilarity_matrix) \
							for i, j in tqdm(list(combinations(range(len(graph_list)), 2)), \
								desc='Generating dissimilarity matrix'))
			else:
				if not USE_MEMMAP:
					with Parallel(n_jobs=n_jobs, prefer='threads', require='sharedmem') as parallel:
						parallel(delayed(get_ged)(i, j, dissimilarity_matrix) \
							for i, j in tqdm(list(combinations_with_replacement(range(0, len(graph_list), CHUNK_SIZE), 2)), \
								desc='Generating dissimilarity matrix'))
				else:
					with Parallel(n_jobs=n_jobs, prefer='threads') as parallel:
						parallel(delayed(get_ged)(i, j, dissimilarity_matrix) \
							for i, j in tqdm(list(combinations_with_replacement(range(0, len(graph_list), CHUNK_SIZE), 2)), \
								desc='Generating dissimilarity matrix'))

			if USE_MEMMAP:
				try:
					shutil.rmtree(folder)
				except:  # noqa
					print('Could not clean-up automatically.')

		if RANDOM_FRAC:
			assert CHUNK_COMPUTE is False
			grid_x, grid_y = np.mgrid[0:len(graph_list), 0:len(graph_list)]
			points = idx_set
			values = [dissimilarity_matrix[idx[0], idx[1]] for idx in idx_set]
			points.extend((i, i) for i in range(len(graph_list)))
			values.extend(0 for i in range(len(graph_list)))

			print('Interpolating matrix')
			dissimilarity_matrix = scipy.interpolate.griddata(points, values, (grid_x, grid_y), 
				method='linear')

		dissimilarity_matrix = np.triu(dissimilarity_matrix, k=1)
		dissimilarity_matrix = dissimilarity_matrix + np.transpose(dissimilarity_matrix)

		assert np.isnan(dissimilarity_matrix).sum() == 0, 'Dissimilarity matrix generated has NaN values'

	with open(DISS_MAT_TEMP, 'wb+') as temp_file:
		print(f'Saving matrix to "{DISS_MAT_TEMP}"')
		np.save(temp_file, dissimilarity_matrix)

	return dissimilarity_matrix