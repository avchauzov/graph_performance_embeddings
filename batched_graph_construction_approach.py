import networkx as nx
import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity


def batched_graph_construction(data, n_neighbors=8, min_cluster_size=4, min_samples=4):
	clustering = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(data)
	labels = clustering.labels_
	
	graphs = []
	for label in np.unique(labels):
		if label == -1:
			continue
		
		cluster_samples = data[labels == label]
		
		similarity_matrix = cosine_similarity(cluster_samples)
		np.fill_diagonal(similarity_matrix, -np.inf)
		
		G = nx.Graph()
		for i in range(similarity_matrix.shape[0]):
			neighbors_index = np.argsort(similarity_matrix[i])[::-1][: n_neighbors]
			
			for j in neighbors_index:
				G.add_edge(i, j, weight=similarity_matrix[i][j])
		
		graphs.append(G)
	
	return graphs


if __name__ == '__main__':
	m, n = 1024, 128
	data = np.random.rand(m, n)
	
	print(batched_graph_construction(data))
