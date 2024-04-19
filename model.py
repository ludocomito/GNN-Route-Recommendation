import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from termcolor import cprint
import torch_geometric
from modules import *

class Model(nn.Module):
	def __init__(self, num_nodes, graph=None, device = "cpu", merging_strategy='cat', args = None, preferences_embedding_dim=128, single_embedding_dim=64, 
                 num_users=180, num_transport_modes=3, embeddings = None, mapping=None, traffic_matrix = None, num_pref_layers=1):
		super(Model, self).__init__()
		self.args = args
		if embeddings is None:
			self.embeddings = nn.Embedding(num_nodes, args.embedding_size)
			self.embeddings = nn.Embedding.from_pretrained(self.embeddings.weight, freeze= not args.trainable_embeddings)
		else:
			self.embeddings = nn.Embedding.from_pretrained(embeddings.float(), freeze= not args.trainable_embeddings)
		
		if (args.gnn is not None):
			cprint("GNN: {}".format(args.gnn),"cyan")
			node_feature_size_for_gnn = self.embeddings.weight.shape[1]
			self.GNN = GNN(
							node_feature_size = node_feature_size_for_gnn,
							output_embedding_size = args.embedding_size, 
							num_layers = args.gnn_layers, 
							hidden_dim = args.hidden_size, 
							graph = graph,
							gnn_type = args.gnn
						)
		
		input_size = 6*args.embedding_size +  preferences_embedding_dim# we need the overall size for concatenated embeddings.
		self.mapping = mapping
		self.device = device

		if args.traffic:
			input_size = 8*args.embedding_size # one for traffic
			self.traffic_matrix = nn.Embedding.from_pretrained(traffic_matrix, freeze = True)	
			self.traffic_linear_initial = nn.Linear(self.traffic_matrix.weight.shape[1], 2*args.embedding_size)	
		if args.attention:
			self.self_attention = nn.MultiheadAttention(2*self.embeddings.weight.shape[1], args.num_heads)

		self.preferences_embedding_model = PreferencesEmbeddingModel(preferences_embedding_dim, single_embedding_dim, 
															   		num_users, num_transport_modes, merging_strategy, num_pref_layers=num_pref_layers, attention=args.attention)

		self.confidence_model = MLP(input_dim = input_size, output_dim = 1, num_layers = args.num_layers, hidden_dim = args.hidden_size) 

	def forward(self, source, dest, nbr, user_ids, transport_modes, timestamps,traffic = None): # (current, dests, next_node, traffic)
		device = self.device
		edge_to_node_mapping = self.mapping
		#print(f'len(source): {len(source)}, len(dest): {len(dest)}, len(nbr): {len(nbr)}')
		#print(f'len(user_ids): {len(user_ids)}, len(transport_modes): {len(transport_modes)}, len(timestamps): {len(timestamps)}')
		#print(user_ids)
		source_left = torch.LongTensor([edge_to_node_mapping[x][0] for x in source]).to(device)
		source_right = torch.LongTensor([edge_to_node_mapping[x][1] for x in source]).to(device) 
		nbr_left = torch.LongTensor([edge_to_node_mapping[x][0] for x in nbr]).to(device)
		nbr_right = torch.LongTensor([edge_to_node_mapping[x][1] for x in nbr]).to(device)
		dest_left = torch.LongTensor([edge_to_node_mapping[x][0] for x in dest]).to(device)
		dest_right = torch.LongTensor([edge_to_node_mapping[x][1] for x in dest]).to(device)

		user_ids = user_ids.to(device)
		transport_modes = transport_modes.to(device)
		timestamps = timestamps.to(device)

		preferences_embedding = self.preferences_embedding_model(user_ids, transport_modes, timestamps)

		if (self.args.gnn is not None):
			self.GNN.data.x = self.embeddings.weight
			embeddings = self.GNN()
		else:
			embeddings = self.embeddings.weight
		# adding an extra row on top
		embeddings = torch.cat((torch.zeros(embeddings.shape[1]).reshape(1,-1).to(self.device),embeddings), dim = 0 )	
		source_vec = torch.cat((embeddings[1+source_left], embeddings[1+source_right]), dim=1) 
		dest_vec = torch.cat((embeddings[1+dest_left], embeddings[1+dest_right]), dim=1)
		nbr_vec = torch.cat((embeddings[1+nbr_left], embeddings[1+nbr_right]), dim=1)

		#print(f'source_vec: {source_vec.shape}, dest_vec: {dest_vec.shape}, nbr_vec: {nbr_vec.shape}, preferences_embedding: {preferences_embedding.shape}')
		
		if self.args.traffic:
			traffic_vec = self.traffic_matrix(torch.LongTensor(traffic).to(device))
			traffic_vec_matched_dim = self.traffic_linear_initial(traffic_vec)

		if not self.args.attention:
			if self.args.traffic:
				x = torch.cat((source_vec, nbr_vec, dest_vec, traffic_vec_matched_dim, preferences_embedding), 1) # concatenating along the last dimension
			else:
				x = torch.cat((source_vec, nbr_vec, dest_vec, preferences_embedding), 1)
		else:
			if self.args.traffic:
				q = torch.stack((source_vec, nbr_vec, dest_vec, traffic_vec_matched_dim, preferences_embedding))
			else:
				q = torch.stack((source_vec, nbr_vec, dest_vec, preferences_embedding))

			z,_ = self.self_attention(q, q, q) 
			x = z.transpose(0,1).reshape(source_vec.shape[0],-1) 

		y = self.confidence_model(x)
		y[nbr_left == -1] = -100 # this is to ensure that the confidence score for the last node is always the lowest
		return y


