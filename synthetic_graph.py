import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx


def generate_graph_without_cycles(num_nodes=100, num_edges=95, num_features=8):
	G = nx.Graph()
	G.add_nodes_from(range(num_nodes))
	
	for node in G.nodes():
		random_features = np.random.rand(num_features)
		G.nodes[node]['features'] = random_features
	
	edge_count = 0
	while edge_count < num_edges:
		a, b = np.random.choice(num_nodes, 2, replace=False)
		
		if not nx.has_path(G, a, b):
			G.add_edge(a, b)
			edge_count += 1
	
	return G


def convert_networkx_to_pyg(graph, feature_key='features'):
	pyg_graph = from_networkx(graph)
	
	features = []
	for _, node_data in graph.nodes(data=True):
		features.append(node_data[feature_key])
	features_tensor = torch.tensor(features, dtype=torch.float)
	
	pyg_graph.x = features_tensor
	return pyg_graph


def visualize_graph(G, with_labels=True, node_color='skyblue', node_size=50, edge_color='k'):
	plt.figure(figsize=(10, 10))
	nx.draw(G, with_labels=with_labels, node_color=node_color, node_size=node_size, edge_color=edge_color)
	plt.title('Graph without Cycles')
	plt.show()


if __name__ == '__main__':
	G = generate_graph_without_cycles()
	visualize_graph(G)
