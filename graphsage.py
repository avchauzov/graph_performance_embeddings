import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv


class GraphSAGENet(torch.nn.Module):
	
	def __init__(self, in_channels, hidden_channels, out_channels):
		super(GraphSAGENet, self).__init__()
		self.conv1 = SAGEConv(in_channels, hidden_channels)
		self.conv2 = SAGEConv(hidden_channels, out_channels)
	
	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		
		x = F.relu(self.conv1(x, edge_index))
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		
		return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGENet(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(200):
	model.train()
	total_loss = 0
	for data in data_loader:
		data = data.to(device)
		optimizer.zero_grad()
		out = model(data)
		loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	print(f'Epoch {epoch}, Loss: {total_loss / len(data_loader)}')
