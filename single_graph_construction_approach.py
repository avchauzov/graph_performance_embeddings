import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def single_graph_construction(data, n_neighbors=8):
	similarity_matrix = cosine_similarity(data)
	np.fill_diagonal(similarity_matrix, -np.inf)
	
	G = nx.Graph()
	G.add_nodes_from(range(len(data)))
	
	for i in range(similarity_matrix.shape[0]):
		neighbors_index = np.argsort(similarity_matrix[i])[::-1][: n_neighbors]
		
		for j in neighbors_index:
			G.add_edge(i, j, weight=similarity_matrix[i][j])
	
	return G


if __name__ == '__main__':
	m, n = 1024, 128
	data = np.random.rand(m, n)
	
	print(single_graph_construction(data))
