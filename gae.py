import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE, GCNConv

from synthetic_graph import convert_networkx_to_pyg


class Encoder(torch.nn.Module):
	
	def __init__(self, in_channels, out_channels):
		super(Encoder, self).__init__()
		self.conv1 = GCNConv(in_channels, 2 * out_channels)
		self.conv2 = GCNConv(2 * out_channels, out_channels)
	
	def forward(self, x, edge_index):
		x = F.relu(self.conv1(x, edge_index))
		x = self.conv2(x, edge_index)
		return x


def train_model(synthetic_graph, out_channels=16, lr=0.01, epochs=128):
	synthetic_graph = convert_networkx_to_pyg(synthetic_graph, feature_key='features')
	num_features = synthetic_graph.x.size(1)
	
	encoder = Encoder(num_features, out_channels)
	model = GAE(encoder)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	
	early_stopping_patience = int(0.1 * epochs)
	early_stopping_counter = 0
	best_loss = float('inf')
	
	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()
		z = model.encode(synthetic_graph.x, synthetic_graph.edge_index)
		loss = model.recon_loss(z, synthetic_graph.edge_index)
		loss.backward()
		optimizer.step()
		
		print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
		
		if loss.item() < best_loss:
			best_loss = loss.item()
			early_stopping_counter = 0
		else:
			early_stopping_counter += 1
		
		if early_stopping_counter >= early_stopping_patience:
			print(f'Early stopping triggered at epoch {epoch + 1}')
			break
