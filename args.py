from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument("-embedding_size", default = 128, type = int)
    parser.add_argument("-hidden_size", default = 256, type = int)
    parser.add_argument("-num_layers", default = 3, type = int)
    parser.add_argument("-gnn", default = "GCN", type = str)
    parser.add_argument("-gnn_layers", default = 1, type = int)
    parser.add_argument("-trainable_embeddings", default = True, type = bool)
    parser.add_argument("-attention", default = False, type = bool)
    parser.add_argument("-loss", default = "v2", type = str)
    parser.add_argument("-num_heads", default = 2, type = int)
    parser.add_argument("-traffic", default = False, type = bool)
    parser.add_argument("-with_dijkstra", default = False, type = bool)
    parser.add_argument("-num_epochs", default = 100, type = int)
    parser.add_argument("-debug_mode", default = False,  action='store_true')
    parser.add_argument("-ignore_unknown_args", default=False, action='store_true')
    parser.add_argument("-run_name", type = str, default='default_run')
    parser.add_argument("-eval_freq", type = int, default=1)
    parser.add_argument("-merging_strategy", type = str, default='cat')
    parser.add_argument("-num_pref_layers", type = int, default=1)
    parser.add_argument("-batch_size", type = int, default=32)

    args, unknown = parser.parse_known_args()
    if len(unknown)!= 0 and not args.ignore_unknown_args:
        print("some unrecognised arguments {}".format(unknown))
        raise SystemExit

    return args

