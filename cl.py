import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_adj

from synthetic_graph import convert_networkx_to_pyg


class Encoder(torch.nn.Module):
	
	def __init__(self, in_channels, out_channels):
		super(Encoder, self).__init__()
		self.conv1 = SAGEConv(in_channels, 2 * out_channels)
		self.conv2 = SAGEConv(2 * out_channels, out_channels)
	
	def forward(self, x, edge_index):
		x = F.relu(self.conv1(x, edge_index))
		x = self.conv2(x, edge_index)
		return x


def generate_graph_views(data, edge_drop_prob=0.25, feature_drop_prob=0.25):
	edge_index1, _ = dropout_adj(data.edge_index, p=edge_drop_prob)
	edge_index2, _ = dropout_adj(data.edge_index, p=edge_drop_prob)
	
	x1 = F.dropout(data.x, p=feature_drop_prob, training=True)
	x2 = F.dropout(data.x, p=feature_drop_prob, training=True)
	
	return Data(x=x1, edge_index=edge_index1), Data(x=x2, edge_index=edge_index2)


def contrastive_loss(z1, z2, temperature=0.075):
	z1_norm = F.normalize(z1, p=2, dim=1)
	z2_norm = F.normalize(z2, p=2, dim=1)
	
	similarity_matrix = torch.mm(z1_norm, z2_norm.t()) / temperature
	targets = torch.arange(z1.size(0)).to(z1.device)
	
	loss = F.cross_entropy(similarity_matrix, targets)
	return loss


def train_model(train, val, out_channels=16, lr=0.01, epochs=128):
	train = convert_networkx_to_pyg(train, feature_key='features')
	val = convert_networkx_to_pyg(val, feature_key='features')
	num_features = train.x.size(1)
	
	model = Encoder(num_features, out_channels)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	
	early_stopping_patience = int(0.1 * epochs)
	early_stopping_counter = 0
	best_val_loss = float('inf')
	
	for epoch in range(epochs):
		view1, view2 = generate_graph_views(train)
		optimizer.zero_grad()
		
		z1 = model(view1.x, view1.edge_index)
		z2 = model(view2.x, view2.edge_index)
		
		loss = contrastive_loss(z1, z2)
		loss.backward()
		optimizer.step()
		
		# Evaluate on the validation graph
		with torch.no_grad():
			view1, view2 = generate_graph_views(val)
			optimizer.zero_grad()
			
			z1 = model(view1.x, view1.edge_index)
			z2 = model(view2.x, view2.edge_index)
			
			val_loss = contrastive_loss(z1, z2)
			
			# Print training progress
			print(f'Epoch: {epoch + 1}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')
			
			# Early stopping logic
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				early_stopping_counter = 0
				# Save the best model
				torch.save(model.state_dict(), 'best_model.pt')
			else:
				early_stopping_counter += 1
			
			if early_stopping_counter >= early_stopping_patience:
				print(f'Early stopping triggered at epoch {epoch + 1}')
				break
	
	# Load the best model
	model.load_state_dict(torch.load('best_model.pt'))
	return model
