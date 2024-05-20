import torch
from sklearn.metrics import f1_score

from utils import EarlyStopping
import numpy as np 
import pickle
import dgl
import torch.nn.functional as F
from tqdm import tqdm
from mcmc_utils import *
import networkx as nx
import scipy.sparse as sp
from scipy import sparse

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluatev2(gamma, new_distr_dictionary, model, features, target, mask, loss_func, augmented_graph_dict, prob_uni):
    
    # renormalize the weights
    prob_uni = 1/len(new_distr_dictionary.keys())
    renorm_probs = dict()
    for k,v in new_distr_dictionary.items():
        if v >= prob_uni:
            renorm_probs[k] = v
    
    renormalization_coeff = sum(list(renorm_probs.values()))
    for k,v in renorm_probs.items():
        renorm_probs[k] = v/renormalization_coeff
    
    weighted_logits = []
    for k,v in renorm_probs.items():
        # key is lambda and v is the probability
        model.eval()
        with torch.no_grad():
            g_instance = augmented_graph_dict[k]
            logits = model(g_instance, features)
            loss = loss_func(logits[mask], target)
            weighted_logits.append(logits*v)

    loss_ce = loss_func(sum(weighted_logits)[mask], target)
    loss = loss_ce
    accuracy, micro_f1, macro_f1 = score(sum(weighted_logits)[mask], target)
    return loss, accuracy, micro_f1, macro_f1



def evaluate(model, g, features, target, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], target)
    accuracy, micro_f1, macro_f1 = score(logits[mask], target)

    return loss, accuracy, micro_f1, macro_f1


def load_data_v2(args):
    with open('data/'+args['dataset']+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open('data/'+args['dataset']+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open('data/'+args['dataset']+'/labels.pkl','rb') as f:
        labels = pickle.load(f)
    num_nodes = edges[0].shape[0]

    # different type of edges have different adjacency matrix
    for i, edge in enumerate(edges):
        if i == 0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    
    # READJUST THE MATRICES
    if torch.equal(A[:,:,0],torch.transpose(A[:,:,1],0,1)):
        print("SAME edge types but different directions, A[:,:,0] == A[:,:,1]")
        A_new = torch.from_numpy(edges[0].todense()+edges[1].todense()).type(torch.FloatTensor).unsqueeze(-1)
    if torch.equal(A[:,:,2],torch.transpose(A[:,:,3],0,1)):
        print("SAME edge types but different directions, A[:,:,2] == A[:,:,3]")
        A_new = torch.cat([A_new, torch.from_numpy(edges[2].todense() + edges[3].todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    num_of_nodes = node_features.size()[0]  # total number of nodes
    total_labelled_nodes = len(labels[0]) + len(labels[1]) + len(labels[2])
    
    train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.LongTensor)
    print('-------------------------------------')
    print('train_node.size()[0]', train_node.size()[0])
    print('valid_node.size()[0]', valid_node.size()[0])
    print('test_node.size()[0]', test_node.size()[0])
    print('total_labelled_nodes', total_labelled_nodes)
    print("total number of nodes", num_of_nodes)
    print('-------------------------------------')
    num_classes = torch.max(train_target).item()+1

    train_node = train_node.to(args['device'])
    valid_node = valid_node.to(args['device'])
    test_node = test_node.to(args['device'])
    train_target = train_target.to(args['device'])
    valid_target = valid_target.to(args['device'])
    test_target = test_target.to(args['device'])


    return A_new, node_features, labels, num_classes, train_node, valid_node, test_node, train_target, \
    valid_target, test_target

def get_adj(args, A, lmd):
    if A.shape[2] == 2:
        new_A = lmd*A[:,:,0] + (1-lmd)*A[:,:,1]
    adj = new_A.cpu().detach().numpy()
    adj = sparse.csr_matrix(adj) 
    adj = normalize(adj + sp.eye(adj.shape[0]))
    g = sparse_mx_to_torch_sparse_tensor(adj)
    g = g.to(args['device'])
    return g
    

def get_empr_dist(args, A, interval, gamma, low, high, features, model, augmented_graph_dict, target, mask, loss_fcn, prev_pt, prev_lmd_list):
    M = 15000
    is_first_lambda = True
    eval_rmse_dict = dict()
    p_lambda_list = []
    n_accept = 0
    for m in tqdm(range(M)):
        if is_first_lambda:
            first_lambda = 0.5
            new_g = get_adj(args, A, first_lambda)
            val_loss, _, _, _ = evaluate(model, new_g, features, target, mask, loss_fcn)
            previous_loss = val_loss
            is_first_lambda = False
            continue
        else:
            new_lambda = proposal_function(low, high, interval)

        # store the graphs for retrieval
        if new_lambda in augmented_graph_dict:
          new_g = augmented_graph_dict[new_lambda]
        else:
          # sum the edge type adjacencies together to get new weighted graphs
          new_g = get_adj(args, A, new_lambda)
          augmented_graph_dict[new_lambda] = new_g

        # store the score for retrieval
        if new_lambda in eval_rmse_dict:
            current_val_loss = eval_rmse_dict[new_lambda]
        else:
            current_val_loss, _, _, _ = evaluate(model, new_g, features, target, mask, loss_fcn)
            eval_rmse_dict[new_lambda] = current_val_loss

        prev_lmb = None  # Not use
        prev_pt = None
        accepted, new_lambda, current_val_loss = MCMC(new_lambda, prev_lmb, gamma, previous_loss, current_val_loss, prev_pt)
        if accepted==1:
            p_lambda_list.append(new_lambda)
            previous_loss = current_val_loss
            # prev_lmb = new_lambda
            n_accept += 1

    return p_lambda_list, augmented_graph_dict, n_accept/M


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
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

def main(args):
    args['dataset'] = args['usedataset']
    A, features, labels, num_classes, train_mask, val_mask, test_mask, train_target, \
    val_target, test_target = load_data_v2(args)
    print("A", type(A), A.shape)
    num_edge_types = A.shape[2]
    print("num_edge_types", num_edge_types)

    interval = 0.05
    g = get_adj(args, A, lmd=0.5) # g is the adj matrix here.
    features = features.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    if args['usedataset'] == "IMDB":
        act = F.relu
    elif args['usedataset'] == "ACM":
        act = torch.nn.LeakyReLU(0.2)
    elif args['usedataset'] == "DBLP":
        act = torch.nn.LeakyReLU(0.1)
    print("act", act)
    from model import GCNv2
    model = GCNv2(nfeat=features.shape[1], 
                  nhid=args['hidden_units'], 
                  nclass=num_classes, 
                  dropout=args['dropout'],
                  layer=args['layer'],
                  activation=act,
                  act_before_dropout=True,
                 ).to(args['device'])
    print(model)

    g = g.to(args['device'])
    augmented_graph_dict = dict()
    gamma = args['gamma']
    low = 0.0
    high = 1.0

    stopper = EarlyStopping(patience=args['patience'], is_homo=False)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    # calculate unifrom probability. this value does not change.
    N = int((high-low)/interval) + 1
    discretize_lambdas = list(np.linspace(low,high,N,endpoint=True))
    prob_uni = 1/len(discretize_lambdas) # this is p0 set to be uniform distrib.
    pO = prob_uni

    prev_pt = None
    prev_lmd_list = None
    for t in range(args['T']):
        if t > 0:
            gamma += 0.5
        # Find Expectation; E-step
        p_lambda_list, augmented_graph_dict, accept_rate = get_empr_dist(args, A, interval, gamma, low, high, 
                                                                         features, model, augmented_graph_dict, 
                                                                         train_target, train_mask, loss_fcn, prev_pt, prev_lmd_list)
        new_distr_dictionary = form_distr_dict(p_lambda_list)
        print("accept_rate", accept_rate)

        for epoch in range(args['Tprime']):
            new_lambda = random.choice(p_lambda_list) # draw according to q(.)
            g_instance = augmented_graph_dict[new_lambda]
            model.train()
            logits = model(g_instance, features)
            loss = loss_fcn(logits[train_mask], train_target)
            pt = new_distr_dictionary[new_lambda]

            # In practice, it can be beneficial to have more flexibity
            # to address overfitting.
            # the coeff can be absored by the lr as long as parity is kept.
            if pt - pO < 0:
                loss = -args['alpha']*loss
            else:
                loss = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], train_target)

            val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluatev2(gamma, new_distr_dictionary, model, features, val_target, val_mask, loss_fcn, augmented_graph_dict, prob_uni)
            early_stop = stopper.step(val_loss.data.item(), val_acc, model)

            if epoch % 20 == 0:
                print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
                    'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
                    epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

            if early_stop:
                break
        if early_stop:
            print("Early stopped")
            break

    stopper.load_checkpoint(model)
    print("new_distr_dictionary", dict(sorted(new_distr_dictionary.items())))
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluatev2(gamma, new_distr_dictionary, model, features, test_target, test_mask, loss_fcn, augmented_graph_dict, prob_uni)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test Acc {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1, test_acc))
    return test_micro_f1, test_macro_f1

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('GAT')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--usedataset', type=str, default="ACM",
                        help='Set dataset')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma or eta starting value')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='alpha to weigh reverse loss')
    parser.add_argument('--T', type=int, default=10,
                        help='T iterations')
    parser.add_argument('--Tprime', type=int, default=25,
                        help='T prime iterations')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='pretrain or not')
    parser.add_argument('--layer', type=int, default=3,
                        help='model layers')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='model layers')
    parser.add_argument('--patience', type=int, default=100,
                        help='early stop patience')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='dropout')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay')
    args = parser.parse_args().__dict__

    args = setup(args)
    print("args", args)
    results_micro = []
    results_macro = []
    for i in range(10):
        test_micro_f1, test_macro_f1 = main(args)
        results_micro.append(test_micro_f1)
        results_macro.append(test_macro_f1)

    import statistics
    print("test micro_f1_ave", sum(results_micro)/len(results_micro), statistics.stdev(results_micro))
    print("test macro_f1_ave", sum(results_macro)/len(results_macro), statistics.stdev(results_macro))
