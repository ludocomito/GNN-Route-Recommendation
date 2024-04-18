import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GNN(torch.nn.Module):
	def __init__(self, node_feature_size, output_embedding_size, num_layers, hidden_dim, graph, gnn_type = "GCN"):
		super(GNN, self).__init__()
		layer_sizes = [node_feature_size] + [hidden_dim]*(num_layers) + [output_embedding_size]

		if gnn_type == "GCN":
			print("Using a GCN")
			layer_function = GCNConv
		elif gnn_type == "GAT":
			print("Using a GAT network")
			layer_function = GATConv
		elif gnn_type == "PGNN":
			print("Using PGNN")
			print("NOT IMPLEMENTED")
			raise SystemExit

		self.layers = nn.ModuleList([layer_function(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])
		self.data = graph # the map graph

	def forward(self):
		x, edge_index = self.data.x, self.data.edge_index

		for i,neighbour_agg in enumerate(self.layers): # simply apply gnn layers + relu for all but last layer
			x = neighbour_agg(x, edge_index)
			if i!= len(self.layers) - 1:
				x = F.relu(x)
		return x
	
class PreferencesEmbeddingModel(nn.Module):
    def __init__(self, preferences_embedding_dim, single_embedding_dim, 
                 num_users, num_transport_modes):
        super(PreferencesEmbeddingModel, self).__init__()

        self.preferences_embedding_dim = preferences_embedding_dim
        self.single_embedding_dim = single_embedding_dim
        self.num_users = num_users
        self.num_transport_modes = num_transport_modes

        self.user_embedding = nn.Embedding(num_users, single_embedding_dim)
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, single_embedding_dim)

        # initialize embeddings using xavier initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.transport_mode_embedding.weight)

        # time embedding model
        self.time_embedding = nn.Linear(6, single_embedding_dim)

        #initilize time embedding using xavier initialization
        nn.init.xavier_uniform_(self.time_embedding.weight)

        # preferences embedding model
        self.preferences_embedding = nn.Linear(3*single_embedding_dim, preferences_embedding_dim)

        # initialize preferences embedding using xavier initialization
        nn.init.xavier_uniform_(self.preferences_embedding.weight)

    def forward(self, user_id, transport_mode, timestamp):
        user_embedded = self.user_embedding(user_id)
        transport_mode_embedded = self.transport_mode_embedding(transport_mode)
        time_embedded = self.time_embedding(timestamp)

        preferences_embedded = torch.cat((user_embedded, transport_mode_embedded, time_embedded), dim=1)
        preferences_embedded = self.preferences_embedding(preferences_embedded)

        return preferences_embedded



class MLP(nn.Module): # given input embeddings, returns one confidence score.
	def __init__(self, input_dim, output_dim, num_layers, hidden_dim):

		assert num_layers >= 0 , "invalid input"
		super(MLP, self).__init__()
		layer_sizes = [input_dim] + [hidden_dim]*(num_layers) + [output_dim]
		self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]) # a list of linear transformations
		self.non_linearity = nn.ReLU()


	def forward(self, x):
		for i,linear_tranform in enumerate(self.layers):
			x = linear_tranform(x)
			if i!= len(self.layers) - 1:
				x = F.relu(x)
		return x