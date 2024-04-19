from args import make_args
from constants import *
import numpy as np
import pickle
from tqdm import tqdm
import torch
from torch import nn 
import pandas as pd
import geopandas as gpd
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from sklearn.cluster import KMeans
from collections import defaultdict
import networkx as nx
import collections
import multiprocessing as mp
from model import Model
import torch_geometric
import numpy as np
import pickle
from tqdm import tqdm
import torch
from torch import nn 
import pandas as pd
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from sklearn.cluster import KMeans
import time
from haversine import haversine
import networkx as nx
import multiprocessing as mp
import torch_geometric
from termcolor import cprint, colored
from collections import OrderedDict
import wandb
import sys

def condense_edges(edge_route):
	global map_edge_id_to_u_v, map_u_v_to_edge_id
	route = [map_u_v_to_edge_id[tuple(map_edge_id_to_u_v[e])] for e in edge_route]
	return route

def fetch_map_fid_to_zero_indexed(data):
	s = set()
	for _,t,_,_ in data:
		s.update(set(t))
	return {el:i for i,el in enumerate(s)}

def relabel_trips(data, mapping):
	return [(idx, [mapping[e] for e in trip], transport_mode,timestamps) for (idx, trip, transport_mode,timestamps) in data]

def remove_loops(path):
	reduced = []
	last_occ = {p:-1 for p in path}
	for i in range(len(path)-1,-1,-1):
		if last_occ[path[i]] == -1:
			last_occ[path[i]] = i
	current = 0
	while(current < len(path)):
		reduced.append(path[current])
		current = last_occ[path[current]] + 1
	return reduced

def nbrs_sanity_check(node_nbrs, data):
	print("SANITY CHECK 1")
	for _, t, _, _ in tqdm(data, dynamic_ncols=True):
		for i in range(len(t)-1):
			assert t[i+1] in node_nbrs[t[i]], "How did this happen?"
	print('Cleared :)')


def create_node_nbrs(forward):  # node neighbours
	start_nodes = defaultdict(set)
	for e in forward:
		u,v = map_edge_id_to_u_v[e]
		start_nodes[u].add(forward[e])
	node_nbrs = {}	# here nodes are actually edges of the road network
	for e in forward:
		_,v = map_edge_id_to_u_v[e]
		node_nbrs[forward[e]] = list(start_nodes[v])
	return node_nbrs

def load_data(less=False, sample=1000, fname='to_edges/train_data_normalized.pkl'):
	print("Loading map matched trajectories")
	f = open(fname, "rb")
	data = pickle.load(f)
	f.close()
			
	if less:
		data = data[:sample]

	data = [(idx, condense_edges(t), transport_mode,timestamps) for (idx, t, transport_mode, timestamps) in tqdm(data, dynamic_ncols=True)]
	
	#if args.remove_loops or args.remove_loops_from_train:
	data = [(idx, remove_loops(t),transport_mode,timestamps) for (idx,t,transport_mode,timestamps) in tqdm(data, dynamic_ncols=True)]
	data = [(idx,t,transport_mode,timestamps) for (idx,t,transport_mode,timestamps) in tqdm(data, dynamic_ncols=True) if len(t) >= 5]	# ignoring very small trips
	
	forward = fetch_map_fid_to_zero_indexed(data)	

	data = relabel_trips(data, forward)
	return data, forward

def load_test_data(forward, less=False, sample = 1000, fname = 'to_edges/val_data_normalized.pkl'):
	print('Loading test/val data')
	f = open(fname, "rb")
	data = pickle.load(f)
	f.close()
	if less:
		data = data[:sample]
	data = [(idx, condense_edges(t), transport_mode,timestamps) for (idx, t, transport_mode, timestamps) in tqdm(data, dynamic_ncols=True)]
	#if args.remove_loops:
	data = [(idx, remove_loops(t), transport_mode,timestamps) for (idx,t, transport_mode,timestamps) in tqdm(data, dynamic_ncols=True)]
	#data = [(idx,t, transport_mode,timestamps) for (idx,transport_mode,t,timestamps) in tqdm(data, dynamic_ncols=True) if len(t) >= 5]	# ignoring very small trips

	orig_num = len(data)
	print('Number of trips in test data (initially): {}'.format(orig_num))
	unseen_data = [trip_tup for trip_tup in data if not set(trip_tup[1]).issubset(forward)]
	for (_, trip, _,_) in unseen_data:
		for e in trip:
			if e not in forward:
				forward[e] = len(forward)

	print('Number of trips with unseen nodes: {}'.format(len(unseen_data)))
	print('Keeping these trips!')
	print('Relabelling trips with updated forward map')
	data = relabel_trips(data, forward)
	return data

def single_source_shortest_path_length_range(graph, node_range, cutoff):
	dists_dict = {}
	for node in node_range:
		dists_dict[node] = nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff, weight='haversine')
	return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


# Get arg
args = make_args()

print(f'Debug_mode: {args.debug_mode}')
if not args.debug_mode:
    print("Initializing wandb")
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="GNN_Route_Recommendation",
        name = args.run_name,
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.001,
        "architecture": args.gnn,
        "epochs": args.num_epochs,
        "hidden_size": args.hidden_size,
        "embedding_size": args.embedding_size,
        "num_layers": args.num_layers,
        "gnn_layers": args.gnn_layers,
        "trainable_embeddings": args.trainable_embeddings,
        "attention": args.attention,
		"merging_strategy": args.merging_strategy,
		"num_pref_layers": args.num_pref_layers
        }
    )

# Load data
edge_df = gpd.read_file(EDGE_DATA)
node_df = gpd.read_file(NODE_DATA)
a = node_df['osmid'].to_numpy()
b = node_df[['y', 'x']].to_numpy()
map_node_osm_to_coords = {e:(u,v) for e,(u,v) in zip(a,b)} # osmid to coords
del a
del b

map_edge_id_to_u_v = edge_df[['u', 'v']].to_numpy()
map_u_v_to_edge_id = {(u,v):i for i,(u,v) in enumerate(map_edge_id_to_u_v)}
unique_edge_labels = list(map_u_v_to_edge_id.values())		# these are from OSM map data, not train data


train_data, forward = load_data(fname=TRAIN_DATA_PATH)
val_data = load_test_data(forward, fname=VAL_DATA_PATH)
test_data = load_test_data(forward, fname=TEST_DATA_PATH)
node_nbrs = create_node_nbrs(forward)

# Perform sanity check
print('Train data sanity check:')
nbrs_sanity_check(node_nbrs, train_data)

print('Val data sanity check:')
nbrs_sanity_check(node_nbrs, val_data)

print('Test data sanity check:')
nbrs_sanity_check(node_nbrs, test_data)

backward = {forward[k]:k for k in forward} 

transformed_graph = nx.DiGraph()
for e1 in node_nbrs:
    for e2 in node_nbrs[e1]:
        if e2 != -1:
            transformed_graph.add_edge(e1, e2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

nodes_used = set()
for e in forward: # for each edge, extract the nodes it connects and add them to the nodes_used set.
    u,v = map_edge_id_to_u_v[e]
    nodes_used.add(u)
    nodes_used.add(v)
nodes_used = list(nodes_used)		
nodes_forward = {node:i for i,node in enumerate(nodes_used)}

# create a mapping from zero-indexed edges to zero-indexed nodes
edge_to_node_mapping = {forward[e]:(nodes_forward[map_edge_id_to_u_v[e][0]], nodes_forward[map_edge_id_to_u_v[e][1]]) for e in forward}
edge_to_node_mapping[-1] = (-1,-1)

f = open('map/graph_with_haversine.pkl','rb')
graph = pickle.load(f)
for e in graph.edges(data=True):
    e[2]['length'] = e[2]['length']/1000 
f.close()

def lipschitz_node_embeddings(nodes_forward, G, k):
	'''
	k is the embedding size.
	'''
	nodes = list(nodes_forward.keys())
	G_temp = G.reverse(copy=True)
	anchor_nodes = random.sample(nodes, k)
	print('Starting Dijkstra')
	num_workers = 12
	cutoff = None
	pool = mp.Pool(processes = num_workers)
	results = [pool.apply_async(single_source_shortest_path_length_range, \
		args=(G_temp, anchor_nodes[int(k/num_workers*i):int(k/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
	output = [p.get() for p in results]
	dists_dict = merge_dicts(output)
	pool.close()
	pool.join()
	print('Dijkstra done')
	embeddings = np.zeros((len(nodes),k))
	for i, node_i in tqdm(enumerate(anchor_nodes), dynamic_ncols=True):
		shortest_dist = dists_dict[node_i]
		for j, node_j in enumerate(nodes):
			dist = shortest_dist.get(node_j, -1)
			if dist!=-1:
				embeddings[nodes_forward[node_j], i] = 1 / (dist + 1)
	embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
	return embeddings

embeddings = None

# graph is the pickled graph (with haversine distances)
print("Generating lipschitz embeddings")
embeddings = lipschitz_node_embeddings(nodes_forward, graph, 128) # lipschitz embeddings
map_node_zero_indexed_to_coords = {nodes_forward[n]:map_node_osm_to_coords[n] for n in nodes_forward} # map_node_osm_to_coords is a dict mapping osm node ids to coordinates

node_embeddings = torch.from_numpy(embeddings).float() if embeddings is not None else None
node_feats = node_embeddings
edge_index = []
for u,v in map_edge_id_to_u_v:
    if u in nodes_forward and v in nodes_forward:
        u, v = nodes_forward[u], nodes_forward[v]
        edge_index.append((u,v))
edge_index = torch.LongTensor(edge_index).T 
torch_graph = torch_geometric.data.Data(x = node_feats, edge_index = edge_index) # instantiate a torch geometric graph with the node features and edge index
torch_graph = torch_graph.to(device)

print(f"Initializing model, merging strategy: {args.merging_strategy}")

model = Model(num_nodes = len(nodes_forward), 
                graph = torch_graph, 
                device = device, 
                args = args, 
                embeddings = node_embeddings, 
                mapping = edge_to_node_mapping,
                traffic_matrix = None,
				merging_strategy=args.merging_strategy,
				num_pref_layers=args.num_pref_layers
            ).to(device)

print("Model initialized")

print("Initializing training parameters")

loss_function_cross_entropy = nn.CrossEntropyLoss(reduction = "sum")
sigmoid_function = nn.Sigmoid()	
optimiser = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

max_nbrs = max(len(nbr_array) for nbr_array in node_nbrs.values()) # max number of neighbours of any node in the graph
num_nodes = len(forward)
for u in range(num_nodes):
    if u in node_nbrs: # if the node has neighbours
        node_nbrs[u].extend([-1]*(max_nbrs - len(node_nbrs[u]))) # pad the node_nbrs with -1s so that all nodes have the same number of neighbours
    else:
        node_nbrs[u] = [-1]*max_nbrs # if the node has no neighbours, pad it with -1s

loss_curve = []
train_acc_curve = []
test_acc_curve = []
max_len = 1 + max(len(t) for _,t,_,_ in train_data)

total_loss = 0
total_trajs = 0
preds = 0
correct = 0
prob_sum = 0

level = 0
val_evals_till_now_reachability = []
val_evals_till_now_precision = []
val_evals_till_now_recall = []

# Evaluation function

JUMP = 1000
def trip_length(path):
	global graph, backward
	return sum([graph[map_edge_id_to_u_v[backward[e]][0]][map_edge_id_to_u_v[backward[e]][1]][0]["length"] for e in path])

def intersections_and_unions(path1, path2):
	global graph, backward
	path1, path2 = set(path1), set(path2)
	intersection = sum([graph[map_edge_id_to_u_v[backward[e]][0]][map_edge_id_to_u_v[backward[e]][1]][0]["length"] for e in path1.intersection(path2)])
	union = sum([graph[map_edge_id_to_u_v[backward[e]][0]][map_edge_id_to_u_v[backward[e]][1]][0]["length"] for e in path1.intersection(path2)])

	return intersection, union

def shorten_path(path, true_dest):
	global map_edge_id_to_u_v, backward, map_node_osm_to_coords
	dest_node = map_edge_id_to_u_v[backward[true_dest]][0]
	_, index = min([(haversine(map_node_osm_to_coords[map_edge_id_to_u_v[backward[edge]][1]], map_node_osm_to_coords[dest_node]), i) for i,edge in enumerate(path)])
	return path[:index+1]

def gen_paths_no_hierarchy(all_paths):
	global JUMP
	ans = []
	for i in tqdm(list(range(0, len(all_paths), JUMP)), desc = "batch_eval", dynamic_ncols=True):
		temp = all_paths[i:i+JUMP]
		#print(f'len temp: {len(temp)}')
		#print(f'temp: {temp}')
		ans.append(gen_paths_no_hierarchy_helper(temp))

	return [t for sublist in ans for t in sublist] # the list of predicted paths

def gen_paths_no_hierarchy_helper(all_paths):
	global model, node_nbrs, max_nbrs, edge_to_node_mapping
	global forward_interval_map
	if args.traffic:
		intervals = [forward_interval_map[(s)] for _,_,(s,_) in all_paths]
	true_paths = [p for _,p,_,_ in all_paths]
	user_ids = torch.stack([uid for uid,_,_,_ in all_paths])
	transport_modes = torch.stack([tm for _,_,tm,_ in all_paths])
	temporal_encodings = torch.stack([te for _,_,_,te in all_paths])
	#print(f'temporal encodings shape: {temporal_encodings.shape}')
	#print(f'user_ids shape: {user_ids.shape})')
	level = 0
	model.eval()
	gens = [[t[0]] for t in true_paths]
	done = set()
	pending = OrderedDict({i:None for i in range(len(all_paths))})
	#print(f'Pending: {pending}')
	with torch.no_grad():
		for _ in tqdm(range(300), desc = "generating trips in lockstep", dynamic_ncols=True):
			true_paths = [all_paths[i][1] for i in pending]
			current_temp = [gens[i][-1] for i in pending]
			current = [c for c in current_temp for _ in node_nbrs[c]]
			pot_next = [nbr for c in current_temp for nbr in node_nbrs[c]] 
			dests = [t[-1] for c,t in zip(current_temp, true_paths) for _ in (node_nbrs[c] if c in node_nbrs else [])]
			#print(f'current: {current}')
   
			# Expand user preferences to match the expanded 'current'
			expanded_user_ids = torch.cat([user_ids[i].repeat(len(node_nbrs[current_temp[i]])) for i in range(len(current_temp))])
			expanded_transport_modes = torch.cat([transport_modes[i].repeat(len(node_nbrs[current_temp[i]])) for i in range(len(current_temp))])
			expanded_temporal_encodings = torch.cat([temporal_encodings[i].unsqueeze(0).repeat(len(node_nbrs[current_temp[i]]), 1) for i in range(len(current_temp))])


			#print(f'user ids shape: {expanded_user_ids.shape}, transport modes shape: {expanded_transport_modes.shape}, temporal encodings shape: {expanded_temporal_encodings.shape}')
			#print(f'expanded tempororal encodings[0]: {expanded_temporal_encodings[5]}')
			traffic = None
			if args.traffic:
				traffic_chosen = [intervals[i] for i in pending]
				traffic = [t for c,t in zip(current_temp, traffic_chosen) for _ in (node_nbrs[c] if c in node_nbrs else [])]
			
			unnormalized_confidence = model(current, dests, pot_next, expanded_user_ids, expanded_transport_modes, expanded_temporal_encodings, traffic)

			chosen = torch.argmax(unnormalized_confidence.reshape(-1, max_nbrs), dim = 1)
			chosen = chosen.detach().cpu().tolist()
			pending_trip_ids = list(pending.keys())
			for identity, choice in zip(pending_trip_ids, chosen):
				choice = node_nbrs[gens[identity][-1]][choice]
				last = gens[identity][-1]
				if choice == -1:
					del pending[identity]
					continue		
				gens[identity].append(choice)
				if choice == all_paths[identity][1][-1]:
					del pending[identity]
			if len(pending) == 0:
				break
			torch.cuda.empty_cache()
	gens = [shorten_path(gen, true[1][-1]) if gen[-1]!=true[1][-1] else gen for gen, true in (zip(gens, all_paths))]
	model.train()
	return gens

def dijkstra(true_trip):
	global args, transformed_graph, max_nbrs, model
	assert args.loss == "v2", "I dont think this will work for loss v1"
	_, (src, *_, dest), (s, _) = true_trip
	g = transformed_graph
	with torch.no_grad():
		current_temp = [c for c in g.nodes()]
		current = [c for c in current_temp for _ in (node_nbrs[c] if c in node_nbrs else []) ]
		pot_next = [nbr for c in current_temp for nbr in (node_nbrs[c] if c in node_nbrs else [])]
		dests = [dest for c in current_temp for _ in (node_nbrs[c] if c in node_nbrs else [])]
		traffic = None
		unnormalized_confidence = model(current, dests, pot_next, traffic)
		unnormalized_confidence = -1*torch.nn.functional.log_softmax(unnormalized_confidence.reshape(-1, max_nbrs), dim = 1)
		transition_nll = unnormalized_confidence.detach().cpu().tolist()
	torch.cuda.empty_cache()
	count = 0
	for u in g.nodes():
		for i,nbr in enumerate(node_nbrs[u]):
			if nbr == -1:
				break
			g[u][nbr]["nll"] = transition_nll[count][i]
		count += 1
	path =  nx.dijkstra_path(g, src, dest, weight = "nll")
	path = [x for x in path]
	return path

def evaluate_no_hierarchy(data, num = 1000, with_correction = False, without_correction = True, with_dijkstra = False):
	global map_node_osm_to_coords, map_edge_id_to_u_v, backward 
	to_do = ["precision", "recall", "reachability", "avg_reachability", "acc", "nll", "generated"]
	results = {s:None for s in to_do}
	cprint("Evaluating {} number of trips".format(num), "magenta")
	partial = random.sample(data, num)
	
	t1 = time.time()
	if with_dijkstra:
		gens = [dijkstra(t) for t in tqdm(partial, desc = "Dijkstra for generation", unit = "trip", dynamic_ncols=True)]
	else:
		gens = gen_paths_no_hierarchy(partial)
	elapsed = time.time() -t1
	results["time"] = elapsed
	jaccs = []
	preserved_with_stamps = partial.copy()
	partial = [p for _,p,_,_ in partial]
	#print("Without correction (everything is weighed according to the edge lengths)")
	generated = list(zip(partial, gens))
	generated = [(t,g) for t,g in generated if len(t)>1]
	lengths = [(trip_length(t), trip_length(g)) for (t,g) in generated]
	inter_union = [intersections_and_unions(t, g) for (t,g) in generated]
	m = len(generated)
	inters = [inter for inter,union in inter_union]
	unions = [union for inter,union in inter_union]
	lengths_gen = [l_g for l_t,l_g in lengths]
	lengths_true = [l_t for l_t,l_g in lengths]
	precs = [i/l if l >0 else 0 for i,l in zip(inters, lengths_gen) ]
	precision1 = round(100*sum(precs)/len(precs), 2)
	recs = [i/l if l >0 else 0 for i,l in zip(inters, lengths_true) ]
	recall1 = round(100*sum(recs)/len(recs), 2)
	deepst_accs = [i/max(l1,l2) for i,l1,l2 in zip(inters, lengths_true, lengths_gen) if max(l1,l2)>0]
	deepst = round(100*sum(deepst_accs)/len(deepst_accs), 2)

	num_reached = len([None for t,g in generated if t[-1] == g[-1]])
	lefts = [haversine(map_node_osm_to_coords[map_edge_id_to_u_v[backward[g[-1]]][0]], map_node_osm_to_coords[map_edge_id_to_u_v[backward[t[-1]]][0]]) for t,g in generated]
	rights = [haversine(map_node_osm_to_coords[map_edge_id_to_u_v[backward[g[-1]]][1]], map_node_osm_to_coords[map_edge_id_to_u_v[backward[t[-1]]][1]]) for t,g in generated]
	reachability = [(l+r)/2 for (l,r) in zip(lefts,rights)]
	all_reach = np.mean(reachability)
	all_reach = round(1000*all_reach,2)
	
	if len(reachability) != num_reached:
		reach_reach = sum(reachability)/(len(reachability)-num_reached)
	else:
		reach_reach = 0

	reach_reach = round(1000*reach_reach,2)

	percent_reached = round(100*(num_reached/len(reachability)), 2)
	print()
	cprint("Precision is                            {}%".format(precision1), "green")
	cprint("Recall is                               {}%".format(recall1), "green")
	print()
	cprint("%age of trips reached is                {}%".format(percent_reached), "green")
	cprint("Avg Reachability(across all trips) is   {}m".format(all_reach), "green")
	cprint("Avg Reach(across trips not reached) is  {}m".format(reach_reach), "green")
	print()
	cprint("Deepst's Accuracy metric is             {}%".format(deepst), "green", attrs = ["dark"])
	print()
	results["precision"] = precision1
	results["reachability"] = percent_reached
	results["avg_reachability"] = (all_reach, reach_reach)
	results["recall"] = recall1
	results["deepst"] = deepst
	results["generated"] = list(zip(preserved_with_stamps, gens))

	return results

def save_model(path_model=CHECKPOINT_PATH, path_extras=CHECKPOINT_SUPPORT_PATH):
	global map_node_osm_to_coords, map_edge_id_to_u_v, forward, model, args
	torch.save(model, path_model + args.run_name+'.pt')
	f = open(path_extras + args.run_name + '.pkl', 'wb')
	pickle.dump((forward, map_node_osm_to_coords, map_edge_id_to_u_v), f)
	f.close()

# Initial evaluation
tqdm.write(colored("\nInitial Eval on Validation set", "blue", attrs = ["bold", "underline"]))
val_results = evaluate_no_hierarchy(data = val_data, num =len(val_data), with_correction = False, with_dijkstra = False)
val_evals_till_now_reachability.append(val_results["reachability"])
val_evals_till_now_precision.append(val_results["precision"])
val_evals_till_now_recall.append(val_results["recall"])

# Training loop
best_f1 = float('-inf')
for epoch in tqdm(range(args.num_epochs), desc = "Epoch", unit="epochs", dynamic_ncols=True):

    random.shuffle(train_data)
    model.train()
    for batch_num,k in tqdm(list(enumerate((range(0, len(train_data), args.batch_size)))), desc = "Batch", unit="steps" ,leave = True, dynamic_ncols=True):
        partial = random.sample(train_data, args.batch_size) 
        valid_trajs = len(partial)

        user_ids = torch.stack([uid for uid,_,_,_ in partial])
        transport_modes = torch.stack([tm for _,_,tm,_ in partial])
        temporal_encodings = torch.stack([te for _,_,_,te in partial])

        next_node = [nbr for _,t,_,_ in partial for i in range(len(t)-1) for nbr in node_nbrs[t[i]]]
        current = [t[i] for _,t,_,_ in partial for i in range(len(t)-1) for _ in node_nbrs[t[i]]]
        dests = [t[-1] for _,t,_,_ in partial for i in range(len(t)-1) for _ in node_nbrs[t[i]]]

        # Prepare indices for expansion matching 'current'
        expanded_user_ids = torch.cat([user_ids[path_idx].repeat(len(node_nbrs[t[node_idx]]), 1) for path_idx, (_, t, _, _) in enumerate(partial) for node_idx in range(len(t)-1)]).squeeze()
        expanded_transport_modes = torch.cat([transport_modes[path_idx].repeat(len(node_nbrs[t[node_idx]]), 1) for path_idx, (_, t, _, _) in enumerate(partial) for node_idx in range(len(t)-1)]).squeeze()
        expanded_temporal_encodings = torch.cat([temporal_encodings[path_idx].unsqueeze(0).repeat(len(node_nbrs[t[node_idx]]), 1) for path_idx, (_, t, _, _) in enumerate(partial) for node_idx in range(len(t)-1)])

        traffic = None

        unnormalized_dist = model(current, dests, next_node, expanded_user_ids, expanded_transport_modes, expanded_temporal_encodings,traffic)

        num_preds = sum(len(t) -1 for _,t,_,_ in partial)	
        true_nbr_class = torch.LongTensor([(node_nbrs[t[i]].index(t[i+1])) for _,t,_,_ in partial for i in range(len(t)-1)]).to(device)
        loss = loss_function_cross_entropy(unnormalized_dist.reshape(-1, max_nbrs), true_nbr_class.to(device))
        preds += num_preds
        preds_in_this_iteration = num_preds
        total_loss += loss.item()
        total_trajs += valid_trajs
        if (valid_trajs > 0):
            if ((batch_num+1)%25==0):
                tqdm.write("Epoch:{}, Batch:{}, loss({}) - per trip: {}, per pred: {}".
                    format(epoch, batch_num+1, 'v2', round(total_loss/total_trajs, 2), round(total_loss/preds, 3)))
                loss_curve.append(total_loss/total_trajs)
                total_loss = 0
                total_trajs = 0
                preds = 0
                correct = 0
                prob_sum = 0
            loss /= valid_trajs
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            torch.cuda.empty_cache()
    if not args.debug_mode:
        wandb.log({"train/loss": loss_curve[-1]})
    if (epoch+1)%args.eval_freq == 0:
        # save_model()
        # cprint('Model saved', 'yellow', attrs=['underline'])
        #tqdm.write(colored("\nDoing a partial evaluation on train set", "blue", attrs = ["bold", "underline"]))
        #tqdm.write(colored("\nStandard",  "cyan", attrs = ["bold", "reverse", "blink"]))
        #train_results =  evaluate_no_hierarchy(data = train_data, 
        #                                        num = min(10000, len(train_data)),
        #                                        with_correction = False, 
        #                                        without_correction = True,
        #                                        with_dijkstra = False)
        #                                        
        #if not args.debug_mode:
        #    wandb.log({"train/train_reachability": train_results["reachability"], "train/train_precision": train_results["precision"], "train/train_recall": train_results["recall"], "train/train_deepst": train_results["deepst"]})
        tqdm.write(colored("\nEvaluation on the validation set (size = {})".format(len(val_data)), "blue", attrs = ["bold", "underline"]))
        tqdm.write(colored("\nStandard",  "cyan", attrs = ["bold", "reverse", "blink"]))
        val_results = evaluate_no_hierarchy(data = val_data, 
                                            num =len(val_data),
                                            with_correction = False,
                                            without_correction = True,
                                            with_dijkstra = False)

        val_precision = val_results["precision"]
        val_recall = val_results["recall"]
        val_f1 = 2*val_precision*val_recall/(val_precision + val_recall)
        val_f1 = round(val_f1, 2)
        if not args.debug_mode:
            wandb.log({"val/val_f1":val_f1, "val/val_reachability": val_results["reachability"], "val/val_precision": val_results["precision"], "val/val_recall": val_results["recall"], "val/val_deepst": val_results["deepst"]})

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_model()
            cprint('Model saved', 'yellow', attrs=['underline'])
			
        val_evals_till_now_reachability.append(val_results["reachability"])
        val_evals_till_now_precision.append(val_results["precision"])
        val_evals_till_now_recall.append(val_results["recall"])
        tqdm.write(colored("Validation Reachability for the previous evals: {}".format(val_evals_till_now_reachability), "yellow"))
        tqdm.write(colored("Validation Precision for the previous evals   : {}".format(val_evals_till_now_precision), "yellow"))
        tqdm.write(colored("Validation Recall for the previous evals      : {}".format(val_evals_till_now_recall), "yellow"))
        print("\n Script currently running is: - \n{}{}\n".format("python -i "," ".join(sys.argv)))
        check_against = 5

# load best model
model = torch.load(CHECKPOINT_PATH + args.run_name + '.pt')
model.eval()

tqdm.write(colored("\nAfter training for {} epochs, ".format(epoch + 1), "yellow"))
tqdm.write(colored("FINAL EVALUATION ON TEST\n", "blue", attrs = ["bold", "underline"]))
tqdm.write(colored("\nStandard",  "cyan", attrs = ["bold", "reverse", "blink"]))
test_results = evaluate_no_hierarchy(data = test_data, 
									num =len(test_data),
									with_correction = True,
									with_dijkstra = False)

reachability = test_results["reachability"]
avg_reach = test_results["avg_reachability"]
precision = test_results["precision"]
recall = test_results["recall"]
deepst_acc = test_results["deepst"]
f1 = 2*precision*recall/(precision + recall)
# round f1 to 2 decimal places
f1 = round(f1, 2)
# Print model stats
stats_file_path = MODEL_STATS_PATH + args.run_name + '.txt'

with open(stats_file_path, "w") as file:
	# Write formatted strings to the file
	file.write(f"{args.run_name.upper()} MODEL STATS\n")
	file.write(f"F1 score is                             {f1}%\n")
	file.write("Precision is                            {}%\n".format(precision))
	file.write("Recall is                               {}%\n".format(recall))
	file.write("\n")  # Writing a newline character to mimic `print()`
	file.write("%age of trips reached is                {}%\n".format(reachability))
	file.write("Avg Reachability(across all trips) is   {}m\n".format(avg_reach[0]))
	file.write("Avg Reach(across trips not reached) is  {}m\n".format(avg_reach[1]))
	file.write("\n")  # Another newline character
	file.write("Deepst's Accuracy metric is             {}%\n".format(deepst_acc))


if not args.debug_mode:
    wandb.log({"test/test_F1": f1,"test/test_reachability": reachability, "test/test_precision": precision, "test/test_recall": recall, "test/test_deepst": deepst_acc})
print("the script that was run here was - \n{}{}".format("python -i "," ".join(sys.argv)))

wandb.finish()