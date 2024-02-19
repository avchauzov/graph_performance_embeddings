from cl import train_model
from synthetic_graph import generate_graph_without_cycles


G = generate_graph_without_cycles()
train_model(synthetic_graph=G, out_channels=16, lr=0.01, epochs=128)
