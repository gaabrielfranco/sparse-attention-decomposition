import torch
import numpy as np
from transformer_lens import HookedTransformer
import pickle
from collections import namedtuple
from ioi_dataset import IOIDataset
import networkx as nx

torch.set_grad_enabled(False)

DB = namedtuple('DB', 'contrib_sv_src contrib_sv_dest sv_signs sv_mags svs_used')
Params = namedtuple('Params', 'attn_thresh perc_contrib_thresh num_prompts')

def contrib_ahs(prompt_id, layer, ah_idx, dest_token, src_token, contrib_sv_dest, contrib_sv_src, svs_used, attn_thresh = 0.7):
    ''' 
    find the upstream attention heads that contribute up to the threshold to this ah's output for these two tokens 
    do this for both the source and the destination tokens
    '''
    # consider all contributing singular vectors
    mat = contrib_sv_dest[prompt_id, layer, ah_idx, dest_token, src_token]
    # return the upstream attention heads that contribute up to the threshold 
    sorted_contribs = np.sort(np.ravel(mat))[::-1]
    thresh = attn_thresh * np.sum(np.ravel(mat))
    cutoff = sorted_contribs[np.where(np.cumsum(sorted_contribs) > thresh)[0][0]]
    dest_contribs = np.where(mat >= cutoff)
    mat = contrib_sv_src[prompt_id, layer, ah_idx, dest_token, src_token]
    sorted_contribs = np.sort(np.ravel(mat))[::-1]
    thresh = attn_thresh * np.sum(np.ravel(mat))
    cutoff = sorted_contribs[np.where(np.cumsum(sorted_contribs) > thresh)[0][0]]
    src_contribs = np.where(mat >= cutoff)
    # return the indices of the upstream attention heads whose contributions meet the threshold criterion
    return src_contribs, dest_contribs

def count_firings(layer, ah_idx, svs_used, ignore_zero_firings = True):
    '''
    count how many times an ah fires
    ignore_zero_firings (= True) suppresses firings where src_token = 0
    '''
    cnt = 0
    for tuple in svs_used.keys():
        (prompt_id, t_layer, t_ah_idx, dest_token, src_token) = tuple
        if ignore_zero_firings:
            if (layer == t_layer) and (ah_idx == t_ah_idx) and (src_token != 0):
                cnt += 1
        elif (layer == t_layer) and (ah_idx == t_ah_idx):
            cnt += 1
    return cnt

def firings(prompt_id, layer, ah_idx, dest_token, svs_used):
    ''' find the src tokens, if any, for which this attention head fires on this destination '''
    return [(prompt_id, layer, ah_idx, dest_token, src_token) for src_token in range(dest_token + 1) 
            if (prompt_id, layer, ah_idx, dest_token, src_token) in svs_used.keys()]

def trace_prompt(contrib_sv_src, contrib_sv_dest, prompt_id, layer, ah_idx, dest_token, src_token, svs_used, 
                   idx_to_gram, gram_to_idx, attn_thresh = 0.7):
    edges = []
    if (prompt_id, layer, ah_idx, dest_token, src_token) not in svs_used.keys():
        return edges
    # we dont trace back from firings when the source token is the start token
    if src_token == 0:
        return edges
    # get upstream ahs that contribute to this firing
    

    src_contrib_ahs, dest_contrib_ahs = contrib_ahs(prompt_id, layer, ah_idx, dest_token, src_token, 
                                                    contrib_sv_dest, contrib_sv_src, svs_used, attn_thresh)
    # code for when individual singular vectors were tracked in full_tracing_data_collection
    # sums over all contributing singular vectors to get total upstream sv input to each of the two tokens
    src_mat = contrib_sv_src[prompt_id, layer, ah_idx, dest_token, src_token]
    dest_mat = contrib_sv_dest[prompt_id, layer, ah_idx, dest_token, src_token]
    # add the edges for each contributing upstream ah to graph
    # both tokens on which (layer, ah_idx) is firing are added to the edge
    # the one that the upstream head is contributing to is identified by 's'/'d'
    for (upstream_layer, upstream_ah_idx) in zip(src_contrib_ahs[0], src_contrib_ahs[1]):
        edges.append((layer, ah_idx, upstream_layer, upstream_ah_idx, src_token, dest_token, src_mat[upstream_layer, upstream_ah_idx], 's'))
    for (upstream_layer, upstream_ah_idx) in zip(dest_contrib_ahs[0], dest_contrib_ahs[1]):
        edges.append((layer, ah_idx, upstream_layer, upstream_ah_idx, src_token, dest_token, dest_mat[upstream_layer, upstream_ah_idx], 'd'))
    # for each upstream ah contributing to the source token
    for (upstream_layer, upstream_ah_idx) in zip(src_contrib_ahs[0], src_contrib_ahs[1]):
        # if it is firing, get its inputs
        for upstream in firings(prompt_id, upstream_layer, upstream_ah_idx, src_token, svs_used):
            (_, _, _, upstream_dest, upstream_src) = upstream
            # and recurse
            edges = edges + trace_prompt(contrib_sv_src, contrib_sv_dest, prompt_id, upstream_layer, upstream_ah_idx, upstream_dest, upstream_src,
                                         svs_used, idx_to_gram, gram_to_idx, attn_thresh)
    # now consider the destination token
    for (upstream_layer, upstream_ah_idx) in zip(dest_contrib_ahs[0], dest_contrib_ahs[1]):
        # if it is firing, get its inputs
        for upstream in firings(prompt_id, upstream_layer, upstream_ah_idx, dest_token, svs_used):
            (_, _, _, upstream_dest, upstream_src) = upstream
            # and recurse
            edges = edges + trace_prompt(contrib_sv_src, contrib_sv_dest, prompt_id, upstream_layer, upstream_ah_idx, upstream_dest, upstream_src,
                                         svs_used, idx_to_gram, gram_to_idx, attn_thresh)
    return edges

def label_edges(prompt_id, edges, idx_to_gram, remove_unlabeled = True):
    ret_edges = []
    for edge in edges:
        (layer, ah_idx, upstream_layer, upstream_ah_idx, src_token, dest_token, mag, type) = edge
        # if remove_unlabeled, we only want to keep the edge if the token that the AH is contributing to can be labeled
        if (type == 's') and ((prompt_id, src_token) in idx_to_gram):
            if (prompt_id, dest_token) in idx_to_gram:
                ret_edges.append((layer, ah_idx, upstream_layer, upstream_ah_idx, 
                                      idx_to_gram[prompt_id, src_token], idx_to_gram[prompt_id, dest_token], mag, type))
            else:
                ret_edges.append((layer, ah_idx, upstream_layer, upstream_ah_idx, 
                                      idx_to_gram[prompt_id, src_token], dest_token, mag, type))
        elif (type == 'd') and ((prompt_id, dest_token) in idx_to_gram):
                if (prompt_id, src_token) in idx_to_gram:
                    ret_edges.append((layer, ah_idx, upstream_layer, upstream_ah_idx, 
                                      idx_to_gram[prompt_id, src_token], idx_to_gram[prompt_id, dest_token], mag, type))
                else:
                    ret_edges.append((layer, ah_idx, upstream_layer, upstream_ah_idx, 
                                      src_token, idx_to_gram[prompt_id, dest_token], mag, type))
        else:
            if remove_unlabeled == False:
                ret_edges.append(edge)
    return ret_edges

def add_new_edges_to_graph(G, edges, prompt_id):
    # corrects for multiple runs from different roots with the same prompt
    for edge in edges:
        (layer, ah_idx, upstream_layer, upstream_ah_idx, src_token, dest_token, mag, type) = edge
        if type == 's':
            token = src_token
        elif type == 'd':
            token = dest_token
        if ((upstream_layer, upstream_ah_idx), (layer, ah_idx, token)) not in G.edges:
            G.add_edge((upstream_layer, upstream_ah_idx), (layer, ah_idx, token), 
                       weight = mag, total = mag, type = type, count = 1,
                       prompt = prompt_id, 
                       src_token = src_token, dest_token = dest_token)
            total_udpate = mag
            count_update = 1
        elif G[(upstream_layer, upstream_ah_idx)][(layer, ah_idx, token)]['prompt'] != prompt_id:
            total_udpate = mag
            count_update = 1
            G[(upstream_layer, upstream_ah_idx)][(layer, ah_idx, token)]['total'] += total_udpate
            G[(upstream_layer, upstream_ah_idx)][(layer, ah_idx, token)]['count'] += count_update
            G[(upstream_layer, upstream_ah_idx)][(layer, ah_idx, token)]['weight'] = (
                G[(upstream_layer, upstream_ah_idx)][(layer, ah_idx, token)]['total'] /
                G[(upstream_layer, upstream_ah_idx)][(layer, ah_idx, token)]['count'])
        else:
            # dont increment weight or count, since this is the same prompt; run from different roots
            total_udpate = 0
            count_update = 0
        if ((layer, ah_idx, token), (layer, ah_idx)) not in G.edges():
            # we set the weights to high arbitrary value b/c we want to see them in graph
            G.add_edge((layer, ah_idx, token), (layer, ah_idx), 
                       weight = 2000, total = 2000, count = 1, type = type,
                      src_token = src_token, dest_token = dest_token)
        else:
            G[(layer, ah_idx, token)][(layer, ah_idx)]['count'] += count_update

def full_tracing_build_graph():
    # Loading the model with no processing (fold_ln, etc.)
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

    # Setting up the dataset
    num_prompts = 256
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N = num_prompts,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=0,
        device=str("cpu")
    )

    # Loading the tracing data
    run_name = f'p{num_prompts}-f1.0-folded-expandedO-scaled'
    fname = f'data/results.nms-{run_name}.pkl'
    with open(fname, 'rb') as fp:
        (db, params) = pickle.load(fp)

    # Unpack the data
    (contrib_sv_src, contrib_sv_dest, sv_signs, sv_mags, svs_used) = db

    # build map from (prompt_id, token) to grammatical role
    gram = ioi_dataset.word_idx.keys()
    idx_to_gram = {}
    gram_to_idx = {}
    for prompt_id in range(len(ioi_dataset)):
        for g in gram:
            idx_to_gram[prompt_id, ioi_dataset.word_idx[g][prompt_id].item()] = g
            gram_to_idx[prompt_id, g] = ioi_dataset.word_idx[g][prompt_id].item()

    # Build the graph
    G = nx.DiGraph()
    for prompt_id in range(num_prompts):
        IO = gram_to_idx[prompt_id, 'IO']
        end = gram_to_idx[prompt_id, 'end']
        t = trace_prompt(contrib_sv_src, contrib_sv_dest, prompt_id, 9, 6, end, IO, svs_used, idx_to_gram, gram_to_idx)
        add_new_edges_to_graph(G, label_edges(prompt_id, t, idx_to_gram), prompt_id)
        t = trace_prompt(contrib_sv_src, contrib_sv_dest, prompt_id, 9, 9, end, IO, svs_used, idx_to_gram, gram_to_idx)
        add_new_edges_to_graph(G, label_edges(prompt_id, t, idx_to_gram), prompt_id)
        t = trace_prompt(contrib_sv_src, contrib_sv_dest, prompt_id, 10, 0, end, IO, svs_used, idx_to_gram, gram_to_idx)
        add_new_edges_to_graph(G, label_edges(prompt_id, t, idx_to_gram), prompt_id)
    for node in G.nodes():
        G.nodes[node]['nfiring'] = count_firings(node[0], node[1], svs_used)

    # Save the graph
    nx.write_graphml(G, f'data/nms-{run_name}.graphml')

    # Build subgraph
    node_subset = [node for node in G.nodes if (G.nodes[node]['nfiring'] > 65) and (node[0] > 6)]
    G_subgraph = nx.subgraph(G, node_subset)
    nx.write_graphml(G_subgraph, f'data/nms-toplayers-{run_name}.graphml')

if __name__ == '__main__':
    full_tracing_build_graph()