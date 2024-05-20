import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from utils import EarlyStopping, set_random_seed

import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl import DropEdge

import scipy.sparse as sp
import random
import pickle
import os

from mcmc_utils import *
from tqdm import tqdm
import statistics


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_edges_to_add(filtered_cos_dict, addprob):
    length_filtered_cos_dict = len(filtered_cos_dict.keys()) 
    num_to_choose = int(length_filtered_cos_dict*addprob)
    edges_to_add = random.sample(filtered_cos_dict.keys(), num_to_choose)
    return edges_to_add


def get_dropedge_graph(g, dropprob):
    transform = DropEdge(p=dropprob)
    dropedge_graph = transform(dgl.remove_self_loop(g).clone())
    return dropedge_graph


def get_new_graph(g, chosen_lambdas, filtered_cos_dict, cuda):
    dropprob = chosen_lambdas[0]
    addprob = chosen_lambdas[1]
    drop_graph = get_dropedge_graph(g, dropprob)
    edges_to_add = get_edges_to_add(filtered_cos_dict, addprob)
    if len(edges_to_add) == 0:
        pass
    else:
        s_list = []
        d_list = []
        for s,d in edges_to_add:
            s_list.append(s)
            d_list.append(d)
        drop_graph.add_edges(s_list, d_list)

    adj = drop_graph.adj(scipy_fmt='csr')
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj.setdiag(0)
    adj = adj.astype(float)

    # set diagonal to zero first before adding sp.eye
    adj = normalize(adj + sp.eye(adj.shape[0]))

    new_adj = sparse_mx_to_torch_sparse_tensor(adj)
    if cuda:
        new_adj = new_adj.cuda()
    return new_adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)) .astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def evaluatev2(model, features, labels, mask, augmented_graph_dict, renorm_probs):
    weighted_logits = []
    for k,v in renorm_probs.items():
        # key is lambda and v is the probability
        model.eval()
        with torch.no_grad():
            g_instance = augmented_graph_dict[k]
            logits = model(g_instance, features)
            weighted_logits.append(logits*v)

    loss_fcn = torch.nn.CrossEntropyLoss()
    labels = labels[mask]
    _, indices = torch.max(sum(weighted_logits)[mask], dim=1)
    correct = torch.sum(indices == labels)
    loss = loss_fcn(sum(weighted_logits)[mask], labels)
    return correct.item() * 1.0 / len(labels), loss


def evaluate(model, g, features, labels, mask):
    loss_fcn = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        loss = loss_fcn(logits, labels)
        return correct.item() * 1.0 / len(labels), loss


def get_empr_dist(g, all_possible_lambda_pairs, filtered_cos_dict, gamma, 
                  features, model, augmented_graph_dict, cuda, labels, val_mask, prev_pt=None):
    if prev_pt is not None:
        candidate_pairs = list(prev_pt.keys())
        all_possible_lambda_pairs = candidate_pairs

    M = 15000
    is_first_lambda = True
    eval_rmse_dict = dict()
    p_lambda_list = []
    n_accept = 0
    for m in tqdm(range(M)):
        if is_first_lambda:
            chosen_lambdas = random.choice(all_possible_lambda_pairs)
            new_adj = get_new_graph(g, chosen_lambdas, filtered_cos_dict, cuda)
            _, val_loss = evaluate(model, new_adj, features, labels, val_mask)
            previous_loss = val_loss
            is_first_lambda = False
            prev_lmb = chosen_lambdas
            continue
        else:
            chosen_lambdas = random.choice(all_possible_lambda_pairs)

        # store the graphs for retrieval
        if chosen_lambdas in augmented_graph_dict:
            new_adj = augmented_graph_dict[chosen_lambdas]
        else:
            # sum the edge type adjacencies together to get new weighted graphs
            new_adj = get_new_graph(g, chosen_lambdas, filtered_cos_dict, cuda)
            augmented_graph_dict[chosen_lambdas] = new_adj

        # store the score for retrieval
        if chosen_lambdas in eval_rmse_dict:
            current_val_loss = eval_rmse_dict[chosen_lambdas]
        else:
            _, current_val_loss = evaluate(model, new_adj, features, labels, val_mask)
            eval_rmse_dict[chosen_lambdas] = current_val_loss

        accepted, new_lambda, current_val_loss = MCMC(chosen_lambdas, prev_lmb, gamma, previous_loss, current_val_loss, prev_pt)
        if accepted==1:
            p_lambda_list.append(new_lambda)
            previous_loss = current_val_loss
            prev_lmb = new_lambda
            n_accept += 1

    BURNIN = 400
    return p_lambda_list[BURNIN:], augmented_graph_dict, n_accept/M


def main(args, filtered_cos_dict):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    n_classes = data.num_labels
    features = g.ndata['feat']
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = normalize(features)
    features = torch.FloatTensor(features).float().cuda()

    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    ################################
    # standard split
    print("standard split")
    num_nodes = g.number_of_nodes()
    node_index = {}
    train_set_index = np.where(train_mask.cpu() == True)[0]

    for i in range(n_classes):
        tmp = [d for d in train_set_index if labels[d].cpu() == i]
        node_index[i] = tmp

    label_n_per_class = args.label_n_per_class
    
    train_indices = []
    for i in range(n_classes):
        print("The training set index for class {} is {}".format(i, node_index[i][0:label_n_per_class]))
        train_indices.extend(node_index[i][0:label_n_per_class])
    print("train_indices", len(train_indices))

    train_mask = torch.from_numpy(sample_mask(train_indices, num_nodes))

    in_feats = features.shape[1]
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    
    candidate_drop_lmd = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
                          0.06, 0.07, 0.08, 0.09, 0.1, 0.11 ,0.12, 0.13, 0.14,0.15,0.16, 0.17, 0.18,0.19,0.2]
    candidate_lmd = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
                     0.06, 0.07, 0.08, 0.09, 0.1, 0.11 ,0.12, 0.13, 0.14,0.15,0.16, 0.17, 0.18,0.19,0.2]

    all_possible_lambda_pairs = [(i, j) for i in candidate_drop_lmd for j in candidate_lmd]
    # tuple the first number is drop and the second is add
    prob_uni = 1/len(all_possible_lambda_pairs) # this is p0 set to be uniform distrib.

    # create GCN model
    from model import GCNv2 
    model = GCNv2(nfeat=in_feats, 
                  nhid=args.n_hidden, 
                  nclass=n_classes, 
                  dropout=args.dropout,
                  layer=args.n_layers,
                  activation=F.relu)

    print("model", model)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameters", count_parameters(model))

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer for GCNv2
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    stopper = EarlyStopping(patience=100)

    idx_val = np.where(val_mask.cpu() == True)[0]
    idx_train = np.where(train_mask.cpu() == True)[0]
    idx_combine = np.concatenate((idx_train, idx_val), axis=0)
    combine_mask = torch.from_numpy(sample_mask(idx_combine, labels.shape[0]))

    # initialize graph
    dur = []
    best_val = 0
    ## MCMC ##
    gamma = 5.0
    augmented_graph_dict = {}

    prev_pt = None
    for epoch in range(args.n_epochs):
        if epoch % 10 == 0:
            if epoch > 0:
                gamma += 10.0
            p_lambda_list, augmented_graph_dict, accept_rate =  get_empr_dist(g, all_possible_lambda_pairs, filtered_cos_dict, 
                                                                              gamma, features, model, augmented_graph_dict, 
                                                                              cuda, labels, combine_mask, prev_pt)
            new_distr_dictionary = form_distr_dict(p_lambda_list)
            print("accept_rate", accept_rate)

            print("new_distr_dictionary", len(new_distr_dictionary.keys()), len(all_possible_lambda_pairs))

            renorm_probs = dict()
            for k,v in new_distr_dictionary.items():
                if v >= prob_uni:
                    renorm_probs[k] = v
            
            renormalization_coeff = sum(list(renorm_probs.values()))
            for k,v in renorm_probs.items():
                renorm_probs[k] = v/renormalization_coeff

        new_lambda = random.choice(p_lambda_list) # draw according to q(.)
        g_instance = augmented_graph_dict[new_lambda]
        pt = new_distr_dictionary[new_lambda]
        pO = prob_uni

        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g_instance, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        if pt - pO < 0:
            loss = -1*args.alpha*loss
        else:
            loss = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc, val_loss = evaluatev2(model, features, labels, val_mask, augmented_graph_dict, renorm_probs)
        # acc, _ = evaluate(model, g_instance, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

        early_stop = stopper.step(val_loss.data.item(), acc, model)
        if early_stop:
            break

    print()
    stopper.load_checkpoint(model)

    p_lambda_list, augmented_graph_dict, accept_rate =  get_empr_dist(g, all_possible_lambda_pairs, filtered_cos_dict, 
                                                                        gamma, features, model, augmented_graph_dict, 
                                                                        cuda, labels, train_mask, prev_pt)
    new_distr_dictionary = form_distr_dict(p_lambda_list)
    renorm_probs = dict()
    for k,v in new_distr_dictionary.items():
        if v >= prob_uni:
            renorm_probs[k] = v
    
    renormalization_coeff = sum(list(renorm_probs.values()))
    for k,v in renorm_probs.items():
        renorm_probs[k] = v/renormalization_coeff

    acc, _ = evaluatev2(model, features, labels, test_mask, augmented_graph_dict, renorm_probs)
    print("Test accuracy {:.2%}".format(acc))
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="alpha for reverse loss")
    parser.add_argument("--value", type=float, default=0.5,
                        help="filter_value")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--label_n_per_class", type=int, default=10,
                        help="data split")
    
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    set_random_seed(args.seed)

    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    src_list, dst_list = g.edges()
    edge_set = set()
    for s, d in zip(src_list, dst_list):
        edge_set.add(tuple([int(s),int(d)]))

    # PRE-PROCESS TO GET THE SET OF CANDIDATE EDGES TO ADD
    if os.path.exists('filtered_' + args.dataset + 'cos_dict.pkl'):
        file = open('filtered_' + args.dataset + 'cos_dict.pkl', 'rb')
        filtered_cos_dict = pickle.load(file)
        file.close()
    else:
        import sys
        cos_dict = {}
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for node1 in range(g.number_of_nodes()):
            for node2 in range(g.number_of_nodes()):
                if node1 != node2 and node1<node2:
                    if args.dataset == 'pubmed':
                        cos_val = cos(features[node1].view(1,-1), features[node2].view(1,-1))
                        if cos_val > 0.4 and cos_val != 1.0:
                            cos_dict[tuple([node1,node2])] = float(cos_val)
                    else:
                        cos_val = cos(features[node1].view(1,-1), features[node2].view(1,-1))
                        cos_dict[tuple([node1,node2])] = float(cos_val)
            if node1 % 50 == 0:
                print('node1', node1, sys.getsizeof(cos_val), sys.getsizeof(cos_dict))
        # open a file, where you want to store the data
        file = open(args.dataset + 'cos_dict.pkl', 'wb')
        pickle.dump(cos_dict, file)
        file.close()

        if args.dataset == 'pubmed':
            filtered_cos_dict = cos_dict
        else:
            filtered_cos_dict = {}
            for item, value in cos_dict.items():
                if value > 0.4 and value != 1.0:
                    filtered_cos_dict[item] = value
        file = open('filtered_' + args.dataset + 'cos_dict.pkl', 'wb')
        pickle.dump(filtered_cos_dict, file)
        file.close()

    filtered_cos_dict_new = {}

    for item, value in filtered_cos_dict.items():
        # if g.has_edges_between([item[0]], [item[1]]) == False and value > args.value:
        if tuple(item) not in edge_set and value > args.value:
            # add edges that were initially not in the graph
            filtered_cos_dict_new[item] = value
    filtered_cos_dict = filtered_cos_dict_new
    print("filtered_cos_dict", len(filtered_cos_dict.keys()))

    result_list = []
    time_list = []
    for i in range(10):
        start_time = time.time()
        test_acc = main(args, filtered_cos_dict)
        result_list.append(test_acc)

        end_time = time.time()
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        time_list.append(elapsed_time)
    
    print("Elapsed time: ", time_list) 
    print("result_list", result_list)
    print("time taken", sum(time_list)/len(time_list), statistics.stdev(time_list))
    print("test accuracy", sum(result_list)/len(result_list), statistics.stdev(result_list))
