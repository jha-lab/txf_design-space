# Graph library class for the design space of transformers

# Author : Shikhar Tuli

import yaml
import numpy as np

import itertools
from tqdm.contrib.itertools import product
from tqdm import tqdm

from joblib import Parallel, delayed
from copy import deepcopy
import json

from utils import graph_util, embedding_util, print_util as pu


PARALLELIZE = False


class GraphLib(object):
	"""Graph Library containing all possible graphs within the design space
	
	Attributes:
		dataset (str): dataset the library is fine-tuned on
		design_space (dict): dictionary of design space hyper-parameters
		hashes_computed (bool): is True is the library generated was checked for isomorphisms 
			in computed graphs. False otherwise, and all Graph objects have their hashes as empty strings
		library (list[Graph]): list of all possible graphs within the design space
		ops_list (list[str]): list of all possible operation blocks in the computational graph
	"""
	def __init__(self, design_space=None, dataset: str = None):
		"""Init GraphLib instance with design_space
		
		Args:
			design_space (str): path to yaml file containing range of hyper-parameters in
				the design space

		Raises:
			AssertionError: if a sanity check fails
		"""
		if design_space:
			with open(design_space) as config_file:
				try:
					config = yaml.safe_load(config_file)
				except yaml.YAMLError as exc:
					print(exc)
				if dataset:
					assert dataset in config.get('datasets'), 'Provided dataset is not supported'	
				self.dataset = dataset
				self.design_space = config.get('architecture')

			# List of all possible operations in the computation graph
			self.ops_list = ['input'] 
			for operation in self.design_space['operation_types']:
				for hidden in self.design_space['hidden_size']:
					for par in self.design_space['operation_parameters'][operation]:
						op = f'{operation}_h{hidden}_p-{par}'
						if op not in self.ops_list: self.ops_list.append(op) 
			for ff in self.design_space['feed-forward_hidden']:
				op = f'f{ff}'
				if op not in self.ops_list: self.ops_list.append(op) 
			self.ops_list.append('add_norm')
			self.ops_list.append('output')

			# Check for no duplicate operations
			assert len(self.ops_list) == len(set(self.ops_list)), \
				f'{pu.bcolors.FAIL}Operations list contains duplicates:\n{self.ops_list}{pu.bcolors.ENDC}'

			# Library of all graphs
			self.library = []

			# Set number of neighbors to 1
			self.num_neighbors = 1
		else:
			self.dataset = ''
			self.design_space = {}
			self.ops_list = []
			self.library = []
			self.num_neighbors = 1

	def __len__(self):
		"""Computes the number of graphs in the library

		Returns:
			len (int): number of graphs in the library
		"""
		return len(self.library)

	def __repr__(self):
		"""Representation of the GraphLib"""
		return f'{pu.bcolors.HEADER}Graph Library with design space:{pu.bcolors.ENDC}\n{self.design_space}' \
			+ f'\n{pu.bcolors.HEADER}Number of graphs:{pu.bcolors.ENDC} {len(self.library)}'

	def build_library(self, 
		layers_per_stack=1, 
		create_graphs=True, 
		check_isomorphism=False, 
		increasing=False, 
		heterogeneous_feed_forward=False):
		"""Build the GraphLib library
		
		Args:
			layers_per_stack (int, optional): the number of layers in the stack
			create_graphs (bool, optional): creates grapha and adds to library
			check_isomorphism (bool, optional): if True, isomorphism is checked 
				for every graph. Default is False, to save compute time.
			increasing (bool, optional): if True, only increasing sizes are considered 
				through the network.
			heterogeneous_feed_forward (bool, optional): if True, feed forward layers are 
				heterogeneous inside each encoder layer, if "nff" for that layer is greater than 1.
		"""
		def _get_stack(lst: list, repeat: int):
			return list(itertools.chain.from_iterable(itertools.repeat(x, repeat) for x in lst))

		print('Creating Graph library')
		count = 0

		assert [layers % layers_per_stack for layers in self.design_space['encoder_layers']] == [0 for layers in \
			self.design_space['encoder_layers']], f'All layers in design space should be divisible by {layers_per_stack}'

		layer_stacks = [layers // layers_per_stack for layers in self.design_space['encoder_layers']]

		for stacks in layer_stacks:
			possible_o = list(itertools.product(self.design_space['operation_types'], repeat=stacks))

			if increasing:
				possible_n = list(itertools.combinations_with_replacement(self.design_space['num_heads'], stacks))
				possible_h = list(itertools.combinations_with_replacement(self.design_space['hidden_size'], stacks))
				if not heterogeneous_feed_forward:
					possible_f = list(itertools.product(self.design_space['feed-forward_hidden'], repeat=stacks))
				else:
					possible_f = [list(itertools.combinations_with_replacement(self.design_space['feed-forward_hidden'], nff)) \
						for nff in self.design_space['number_of_feed-forward_stacks']]
				possible_nff = list(itertools.combinations_with_replacement(
										self.design_space['number_of_feed-forward_stacks'], stacks))
			else:
				possible_n = list(itertools.product(self.design_space['num_heads'], repeat=stacks))
				possible_h = list(itertools.product(self.design_space['hidden_size'], repeat=stacks))
				if not heterogeneous_feed_forward:
					possible_f = list(itertools.product(self.design_space['feed-forward_hidden'], repeat=stacks))
				else:
					possible_f = [list(itertools.product(self.design_space['feed-forward_hidden'], repeat=nff)) \
						for nff in self.design_space['number_of_feed-forward_stacks']]
				possible_nff = list(itertools.product(
										self.design_space['number_of_feed-forward_stacks'], repeat=stacks))

			def create_graph(self, n, h, o, nff, count):
				possible_p = itertools.product(*[self.design_space['operation_parameters'][o[stack]] \
					for stack in range(stacks)])
				if not create_graphs:
					if not heterogeneous_feed_forward:
						count += len(list(possible_p)) * len(list(possible_f))
					else:
						count += len(list(possible_p)) * len(list(itertools.product(* \
							[possible_f[self.design_space['number_of_feed-forward_stacks'].index(nff[stack])] \
							for stack in range(stacks)])))
					return count
				for p in possible_p:
					if not heterogeneous_feed_forward:
						for f in possible_f:
							model_dict = {'l': stacks * layers_per_stack, 'h': _get_stack(list(h), layers_per_stack), \
								'n': _get_stack(list(n), layers_per_stack), 'o': _get_stack(list(o), layers_per_stack), \
								'f': _get_stack([[f[stack]] *  nff[stack] for stack in range(stacks)], layers_per_stack), \
								'p': _get_stack(list(p), layers_per_stack)}
							new_graph = Graph(model_dict, self.ops_list, compute_hash=True)
							count += 1
							if check_isomorphism:
								assert new_graph.hash not in [graph.hash for graph in self.library], \
									f'{pu.bcolors.FAIL}Two graphs found with same hash! ' \
									+ f'Check if they are isomorphic:{pu.bcolors.ENDC}\n' \
									+ f'Graph-1: {new_graph.model_dict}\n' \
									+ f'Graph-2: {graph.model_dict for graph in self.library if graph.hash == new_graph.hash}'
							self.library.append(new_graph)
					else:
						for f in itertools.product(*[possible_f[nff[stack]-1] for stack in range(stacks)]):
							model_dict = {'l': stacks * layers_per_stack, 'h': _get_stack(list(h), layers_per_stack), \
								'n': _get_stack(list(n), layers_per_stack), 'o': _get_stack(list(o), layers_per_stack), \
								'f': _get_stack(list(f), layers_per_stack), \
								'p': _get_stack(list(p), layers_per_stack)}
							new_graph = Graph(model_dict, self.dataset, self.ops_list, compute_hash=True)
							count += 1
							if check_isomorphism:
								assert new_graph.hash not in [graph.hash for graph in self.library], \
									f'{pu.bcolors.FAIL}Two graphs found with same hash! ' \
									+ f'Check if they are isomorphic:{pu.bcolors.ENDC}\n' \
									+ f'Graph-1: {new_graph.model_dict}\n' \
									+ f'Graph-2: {graph.model_dict for graph in self.library if graph.hash == new_graph.hash}'
							self.library.append(new_graph)

				return count

			if not PARALLELIZE:
				for n, h, o, nff in product(possible_n, possible_h, possible_o, possible_nff, \
					desc=f'Generating transformers with {stacks * layers_per_stack} encoder layers'):
					count = create_graph(self, n, h, o, nff, count)
					# print(f'\r{pu.human_format(count)}', end='')
			else:
				with Parallel(n_jobs=8, prefer='threads', require='sharedmem') as parallel:
					parallel(delayed(create_graph)(self, n, h, o, nff, count) \
						for n, h, o, nff in product(possible_n, possible_h, possible_o, possible_nff, \
							desc=f'Generating transformers with {stacks * layers_per_stack} encoder layers'))

		print(f'{pu.bcolors.OKGREEN}Graphs created{": " + str(count) if not PARALLELIZE else ""}!{pu.bcolors.ENDC} ' \
			+ f'\n{len(self.library)} graphs within the design space in the library.')

	def interpolate_neighbors(self, 
		graph1: 'Graph', 
		graph2: 'Graph', 
		old_layers_per_stack: int, 
		new_layers_per_stack: int,
		check_isomorphism=False, 
		increasing=False,
		heterogeneous_feed_forward=False):
		"""Interpolates between two neighbors with finer grained stacks

		Args:
			graph1 (Graph): first graph in the library
			graph2 (Graph): second graph in the library
			old_layers_per_stack (int): old layers per stack
			new_layers_per_stack (int): new layers per stack
			check_isomorphism (bool, optional): if True, isomorphism is checked 
				for every graph. Default is False, to save compute time.
			increasing (bool, optional): if True, only increasing sizes are considered 
				through the network.
			heterogeneous_feed_forward (bool, optional): if True, feed forward layers are 
				heterogeneous inside each encoder layer, if "nff" for that layer is greater than 1.

		Returns:
			interpolants (list): list of Graph objects between graph1 and graph2
		"""
		assert new_layers_per_stack <= old_layers_per_stack and old_layers_per_stack % new_layers_per_stack == 0, \
			'Old number of layers per stack should be divisible by new number of layers per stack'

		interpolant = []

		stack_mult = old_layers_per_stack // new_layers_per_stack

		smaller_length = min(graph1.model_dict['l'], graph2.model_dict['l'])
		larger_length = max(graph1.model_dict['l'], graph2.model_dict['l'])
		different_lengths = smaller_length != larger_length
		call_different = False

		stack_design_spaces = []

		possible_o_stack = []
		possible_n_stack = []
		possible_f_stack = []
		possible_h_stack = []
		possible_nff_stack = []

		for stack in range(smaller_length // old_layers_per_stack):
			# Create a local design space of the course stack
			stack_design_space = {}

			stack_design_space['hidden_size'] = list(set([graph.model_dict['h'][stack * old_layers_per_stack] \
				for graph in [graph1, graph2]]))

			stack_design_space['num_heads'] = list(set([graph.model_dict['n'][stack * old_layers_per_stack] \
				for graph in [graph1, graph2]]))

			stack_design_space['operation_types'] = list(set([graph.model_dict['o'][stack * old_layers_per_stack] \
				for graph in [graph1, graph2]]))

			stack_design_space['number_of_feed-forward_stacks'] = list(set( \
				[len(graph.model_dict['f'][stack * old_layers_per_stack]) for graph in [graph1, graph2]]))

			stack_design_space['feed-forward_hidden'] = list(set([graph.model_dict['f'][stack * old_layers_per_stack][0] \
				for graph in [graph1, graph2]]))

			stack_design_space['operation_parameters'] = {op: [] for op in stack_design_space['operation_types']}
			operation_parameters = [graph.model_dict['p'][stack * old_layers_per_stack] for graph in [graph1, graph2]]
			for par in operation_parameters:
				op = [op for op in stack_design_space['operation_types'] \
					if par in self.design_space['operation_parameters'][op]]
				assert len(op) == 1
				op = op[0]
				stack_design_space['operation_parameters'][op].append(par)

			stack_design_spaces.extend([stack_design_space] * stack_mult)

			possible_o = list(itertools.product(stack_design_space['operation_types'], repeat=stack_mult))

			if increasing:
				possible_n = list(itertools.combinations_with_replacement(stack_design_space['num_heads'], stack_mult))
				possible_h = list(itertools.combinations_with_replacement(stack_design_space['hidden_size'], stack_mult))
				if not heterogeneous_feed_forward:
					possible_f = list(itertools.product(stack_design_space['feed-forward_hidden'], repeat=stack_mult))
				else:
					possible_f = [list(itertools.combinations_with_replacement(stack_design_space['feed-forward_hidden'], \
						nff)) for nff in stack_design_space['number_of_feed-forward_stacks']]
				possible_nff = list(itertools.combinations_with_replacement(
										stack_design_space['number_of_feed-forward_stacks'], stack_mult))
			else:
				possible_n = list(itertools.product(stack_design_space['num_heads'], repeat=stack_mult))
				possible_h = list(itertools.product(stack_design_space['hidden_size'], repeat=stack_mult))
				if not heterogeneous_feed_forward:
					possible_f = list(itertools.product(stack_design_space['feed-forward_hidden'], repeat=stack_mult))
				else:
					possible_f = [list(itertools.product(stack_design_space['feed-forward_hidden'], repeat=nff)) \
						for nff in stack_design_space['number_of_feed-forward_stacks']]
				possible_nff = list(itertools.product(
										stack_design_space['number_of_feed-forward_stacks'], repeat=stack_mult))

			possible_o_stack.append(possible_o)
			possible_n_stack.append(possible_n)
			possible_h_stack.append(possible_h)
			possible_f_stack.append(possible_f)
			possible_nff_stack.append(possible_nff)

		possible_o = list(list(itertools.chain(*o)) for o in itertools.product(*possible_o_stack))
		possible_n = list(list(itertools.chain(*n)) for n in itertools.product(*possible_n_stack))
		possible_h = list(list(itertools.chain(*h)) for h in itertools.product(*possible_h_stack))
		possible_f = list(list(itertools.chain(*f)) for f in itertools.product(*possible_f_stack))
		possible_nff = list(list(itertools.chain(*nff)) for nff in itertools.product(*possible_nff_stack))

		stacks = smaller_length // new_layers_per_stack
		count = 0

		def _get_stack(lst: list, repeat: int):
			return list(itertools.chain.from_iterable(itertools.repeat(x, repeat) for x in lst))

		def create_graph(interpolants, n, h, o, nff, count):
			possible_p = itertools.product(*[stack_design_spaces[stack]['operation_parameters'][o[stack]] \
				for stack in range(smaller_length // new_layers_per_stack)])
			
			if call_different:
				possible_p = list(list(itertools.chain(*p)) for p in itertools.product(*[possible_p, \
				[larger_graph.model_dict['p'][smaller_length:larger_length]]]))

			for p in possible_p:
				if not heterogeneous_feed_forward:
					for f in possible_f:
						model_dict = {'l': stacks * new_layers_per_stack, 'h': _get_stack(list(h), new_layers_per_stack), \
							'n': _get_stack(list(n), new_layers_per_stack), 'o': _get_stack(list(o), new_layers_per_stack), \
							'f': _get_stack([[f[stack]] *  nff[stack] for stack in range(stacks)], new_layers_per_stack), \
							'p': _get_stack(list(p), new_layers_per_stack)}
						new_graph = Graph(model_dict, self.ops_list, compute_hash=True)
						count += 1
						if check_isomorphism:
							assert new_graph.hash not in [graph.hash for graph in self.library], \
								f'{pu.bcolors.FAIL}Two graphs found with same hash! ' \
								+ f'Check if they are isomorphic:{pu.bcolors.ENDC}\n' \
								+ f'Graph-1: {new_graph.model_dict}\n' \
								+ f'Graph-2: {graph.model_dict for graph in self.library if graph.hash == new_graph.hash}'
						interpolants.append(new_graph)
				else:
					for f in itertools.product(*[possible_f[ \
							stack_design_space['number_of_feed-forward_stacks'].index(nff[stack])] for stack in range(stacks)]):
						model_dict = {'l': stacks * new_layers_per_stack, 'h': _get_stack(list(h), new_layers_per_stack), \
							'n': _get_stack(list(n), new_layers_per_stack), 'o': _get_stack(list(o), new_layers_per_stack), \
							'f': _get_stack(list(f), new_layers_per_stack), \
							'p': _get_stack(list(p), new_layers_per_stack)}
						new_graph = Graph(model_dict, self.ops_list, compute_hash=True)
						count += 1
						if check_isomorphism:
							assert new_graph.hash not in [graph.hash for graph in self.library], \
								f'{pu.bcolors.FAIL}Two graphs found with same hash! ' \
								+ f'Check if they are isomorphic:{pu.bcolors.ENDC}\n' \
								+ f'Graph-1: {new_graph.model_dict}\n' \
								+ f'Graph-2: {graph.model_dict for graph in self.library if graph.hash == new_graph.hash}'
						interpolants.append(new_graph)

			return count

		interpolants = []

		if not PARALLELIZE:
			for n, h, o, nff in product(possible_n, possible_h, possible_o, possible_nff, \
				desc=f'Generating transformers with {stacks * new_layers_per_stack} encoder layers'):
				count = create_graph(interpolants, n, h, o, nff, count)
				# print(f'\r{pu.human_format(count)}', end='')
		else:
			with Parallel(n_jobs=8, prefer='threads', require='sharedmem') as parallel:
				parallel(delayed(create_graph)(interpolants, n, h, o, nff, count) \
					for n, h, o, nff in product(possible_n, possible_h, possible_o, possible_nff, \
						desc=f'Generating transformers with {stacks * new_layers_per_stack} encoder layers'))

		if different_lengths:
			larger_graph = graph1 if graph1.model_dict['l'] == larger_length else graph2
			call_different = True

			possible_o = list(list(itertools.chain(*o)) for o in itertools.product(*[possible_o, \
				[larger_graph.model_dict['o'][smaller_length:larger_length]]]))
			possible_n = list(list(itertools.chain(*n)) for n in itertools.product(*[possible_n, \
				[larger_graph.model_dict['n'][smaller_length:larger_length]]]))
			possible_h = list(list(itertools.chain(*h)) for h in itertools.product(*[possible_h, \
				[larger_graph.model_dict['h'][smaller_length:larger_length]]]))
			possible_f = list(list(itertools.chain(*f)) for f in itertools.product(*[possible_f, \
				larger_graph.model_dict['f'][smaller_length:larger_length]]))
			possible_nff = list(list(itertools.chain(*nff)) for nff in itertools.product(*[possible_nff, \
				[[len(larger_graph.model_dict['f'][i]) for i in range(smaller_length, larger_length)]]]))

			stacks = larger_length // new_layers_per_stack

			if not PARALLELIZE:
				for n, h, o, nff in product(possible_n, possible_h, possible_o, possible_nff, \
					desc=f'Generating transformers with {stacks * new_layers_per_stack} encoder layers'):
					count = create_graph(interpolants, n, h, o, nff, count)
					# print(f'\r{pu.human_format(count)}', end='')
			else:
				with Parallel(n_jobs=8, prefer='threads', require='sharedmem') as parallel:
					parallel(delayed(create_graph)(interpolants, n, h, o, nff, count) \
						for n, h, o, nff in product(possible_n, possible_h, possible_o, possible_nff, \
							desc=f'Generating transformers with {stacks * new_layers_per_stack} encoder layers'))

		return interpolants

	def build_embeddings(self, embedding_size: int, algo='MDS', kernel='WeisfeilerLehman', diss_mat_file: str = None, 
		neighbors=100, n_jobs=8):
		"""Build the embeddings of all Graphs in GraphLib using MDS
		
		Args:
			embedding_size (int): size of the embedding
			algo (str): algorithm to use for generating embeddings. Can be any
				of the following:
					- 'MDS'
					- 'GD'
			kernel (str, optional): the kernel to be used for computing the dissimilarity 
				matrix. Can be any of the following:
					- 'WeisfeilerLehman'
					- 'NeighborhoodHash'
					- 'RandomWalkLabeled'
					- 'GraphEditDistance'
				The default value is 'WeisfeilerLehman'
			diss_mat_file (str, optional): path to the dissimilarity matrix file. If given, 
				dissimilarity matrix is not computed
			neighbors (int, optional): number of nearest neighbors to save for every graph
			n_jobs (int, optional): number of parrallel jobs for joblib
		"""
		print('Building embeddings for the Graph library')

		# Create list of graphs (tuples of adjacency matrices and ops)
		graph_list = [self.library[i].graph for i in range(len(self))]

		if diss_mat_file is None:
			# Generate dissimilarity_matrix using the specified kernel
			diss_mat = graph_util.generate_dissimilarity_matrix(graph_list, kernel=kernel, ops_list=self.ops_list, n_jobs=n_jobs)
		else:
			diss_mat = np.load(open(diss_mat_file, 'rb'))

		# Generate embeddings using MDS or GD
		if algo == 'MDS':
			embeddings = embedding_util.generate_mds_embeddings(diss_mat, embedding_size=embedding_size, n_jobs=n_jobs)
		else:
			embeddings = embedding_util.generate_grad_embeddings(diss_mat, 
				embedding_size=embedding_size, n_jobs=n_jobs, silent=True)

		# Get neighboring graph in the embedding space, for all Graphs
		neighbor_idx = embedding_util.get_neighbors(embeddings, 'biased', graph_list, neighbors)

		# Update embeddings and neighbors of all Graphs in GraphLib
		for i in range(len(self)):
			self.library[i].embedding = embeddings[i, :]
			self.library[i].neighbors = [self.library[int(neighbor_idx[i, n])].hash for n in range(neighbors)]

		self.num_neighbors = neighbors

		print(f'{pu.bcolors.OKGREEN}Embeddings generated, of size: {embedding_size}{pu.bcolors.ENDC}')

	def build_naive_embeddings(self, neighbors=100, compute_zscore=True):
		"""Build the embeddings of all Graphs in GraphLib naively. Only one neighbor
		is saved for each graph
		"""
		print('Building embeddings for the Graph library')

		# Create list of model dictionaries
		model_dict_list = [graph.model_dict for graph in self.library]

		# Generate naive embeddings
		embeddings = embedding_util.generate_naive_embeddings(model_dict_list, self.design_space, compute_zscore=compute_zscore)

		# Get neighboring graph in the embedding space, for all Graphs
		neighbor_idx = embedding_util.get_neighbors(embeddings, 'distance', [graph.graph for graph in self.library], neighbors)

		# Update embeddings and neighbors of all Graphs in GraphLib
		for i in range(len(self)):
			self.library[i].embedding = embeddings[i, :]
			self.library[i].neighbor = [self.library[int(neighbor_idx[i, n])].hash for n in range(neighbors)]

		self.num_neighbors = neighbors

		print(f'{pu.bcolors.OKGREEN}Embeddings generated, of size: {embeddings.shape[1]}{pu.bcolors.ENDC}')

	def get_graph(self, model_hash=None, model_dict=None) -> 'Graph':
		"""Return a Graph object in the library from hash
		
		Args:
			model_hash (str, optional): hash of the graph in
				the library
			model_dict (dict, optional): model_dict of the graph to be
				loaded
		
		Returns:
			Graph object, model index
		
		Raises:
			ValueError: if neither model_hash nor model_dict are provided
		"""
		if model_hash is not None:
			if type(model_hash) != str:
				raise ValueError('Dictionary provided for model_hash. Use keyword argument')
			hashes = [graph.hash for graph in self.library]
			model_idx = hashes.index(model_hash)
			return self.library[model_idx], model_idx
		elif model_dict is not None:
			if type(model_dict) != dict:
				raise ValueError('String provided for model_dict. Use keyword argument')
			model_dicts = [graph.model_dict for graph in self.library]
			model_idx = model_dicts.index(model_dict)
			return self.library[model_idx], model_idx
		else:
			raise ValueError('Neither model_hash nor model_dict was provided')

	def save_dataset(self, file_path: str):
		"""Saves dataset of all transformers in the design space
		
		Args:
			file_path (str): file path to save dataset
		"""
		if self.library and self.library[0].embedding is not None:
			embeddings_list = [graph.embedding.tolist() for graph in self.library]
		else:
			embeddings_list = [None for graph in self.library]

		with open(file_path, 'w', encoding ='utf8') as json_file:
			json.dump({'dataset': self.dataset,
						'design_space': self.design_space,
						'ops_list': self.ops_list,
						'num_neighbors': self.num_neighbors, 
						'model_dicts': [graph.model_dict for graph in self.library],
						'hashes': [graph.hash for graph in self.library],
						'embeddings': embeddings_list,
						'neighbors': [graph.neighbors for graph in self.library],
						'performances': [graph.performance for graph in self.library]}, 
						json_file, ensure_ascii = True)

		print(f'{pu.bcolors.OKGREEN}Dataset saved to:{pu.bcolors.ENDC} {file_path}')

	@staticmethod
	def load_from_dataset(file_path: str) -> 'GraphLib':
		"""Summary
		
		Args:
			file_path (str): file path to load dataset
		
		Returns:
			GraphLib: a GraphLib object
		"""
		graphLib = GraphLib()

		with open(file_path, 'r', encoding ='utf8') as json_file:
			dataset_dict = json.load(json_file)

			graphLib.dataset = dataset_dict.get('dataset', '')
			graphLib.ops_list = dataset_dict['ops_list']
			graphLib.design_space = dataset_dict['design_space']
			graphLib.num_neighbors = dataset_dict['num_neighbors']

			if dataset_dict['embeddings'][0] is not None:
				embeddings_list = [np.array(embedding) for embedding in dataset_dict['embeddings']]
			else:
				embeddings_list = [None for embedding in dataset_dict['embeddings']]
			
			for i in range(len(dataset_dict['model_dicts'])):
				graph = Graph(dataset_dict['model_dicts'][i], graphLib.ops_list, compute_hash=False)
				graph.hash = dataset_dict['hashes'][i]
				graph.embedding = embeddings_list[i]
				graph.neighbors = dataset_dict['neighbors'][i]
				graph.performance = dataset_dict['performances'][i]

				graphLib.library.append(graph)

		return graphLib


class Graph(object):
	"""Graph class to represent a computational graph in the design space
	
	Attributes:
		performance (float): performance of the model for the given dataset
		embedding (np.ndarray): embedding for every graph in the design space
		graph (tuple(np.ndarray, list[str])): model graph as a tuple of adjacency matrix and 
			a list of operations
		hash (str): hash for current graph to check isomorphism
		model_dict (dict): dictionary of model hyper-parameter choices for this graph
		neighbors (list[str]): hashes of the nearest neighbors for this graph in order of
			nearest to farther neighbors
		ops_idx (list[int]): list of operation indices
	"""
	def __init__(self, model_dict: dict, ops_list: list, compute_hash: bool, hash_algo='sha256'):
		"""Init a Graph instance from model_dict
		
		Args:
			model_dict (dict): dictionary with the hyper-parameters for current 
				model graph
			ops_list (list): list of all possible operation blocks in 
				the computational graph
			compute_hash (bool): if True, hash is computed, else, is None.
			hash_algo (str, optional): hash algorithm to use among ["md5", "sha256", "sha512"]
		"""
		self.model_dict = model_dict

		# Generate model graph
		self.graph = graph_util.model_dict_to_graph(model_dict, ops_list)

		# Generate hash to check isomorphic graphs
		if compute_hash: 
			self.hash = graph_util.hash_graph(*self.graph, algo=hash_algo)
		else:
			self.hash = None

		# Keep operation indices for similarity metrics
		if ops_list:
			self.ops_idx = [ops_list.index(op) for op in self.graph[1]]
		else:
			self.ops_idx = None

		# Initialize embedding
		self.embedding = None

		# Initialize the nearest neighboring graph
		self.neighbors = None

		# Initialize performance metric
		self.performance = None

	def __repr__(self):
		"""Representation of the Graph"""
		return f'{pu.bcolors.HEADER}Graph model_dict:{pu.bcolors.ENDC} {self.model_dict}\n' \
			+ f'{pu.bcolors.HEADER}Accuracy:{pu.bcolors.ENDC} {self.performance}\n' \
			+ f'{pu.bcolors.HEADER}Embedding:{pu.bcolors.ENDC} {self.embedding}\n' \
			+ f'{pu.bcolors.HEADER}Hash:{pu.bcolors.ENDC} {self.hash}'
