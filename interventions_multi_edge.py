import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from collections import namedtuple
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
import networkx as nx

from transformer_lens import HookedTransformer
from ioi_dataset import IOIDataset
from utils import compute_projections, compute_projections_random, get_omega_decomposition_all_ahs, projection_intervention

DB = namedtuple('DB', 'contrib_sv_src contrib_sv_dest sv_signs sv_mags svs_used')
Params = namedtuple('Params', 'attn_thresh perc_contrib_thresh num_prompts')

random = np.random.RandomState(0) # Global random state for reproducibility

torch.set_grad_enabled(False)

def run_intervention_B(model, cache, ioi_dataset, intervention_dict, intervention_type, is_boosting):    
    def intervention_fn_b(x, hook, ah_idx, interv_value):
        print(f"Intervention B in {hook.name}, ah_idx={ah_idx}")
        x[:, :, ah_idx, :] = interv_value
        return x
    
    print(f"Running intervention with intervention_type={intervention_type} with is_boosting={is_boosting}")

    # Reset hooks
    model.reset_hooks(including_permanent=True)

    attn_input_interv_map = {}

    for (layer, ah_idx) in intervention_dict:
        print(f"Intervening in layer={layer}, head={ah_idx}")
        # Getting the attention input (it's the same for all heads)
        attn_input = cache[f"blocks.{layer}.ln1.hook_normalized"]
    
        # Intervening in the attention input
        attn_input_interv = deepcopy(attn_input)
        if is_boosting:
            attn_input_interv += intervention_dict[(layer, ah_idx)]
        else:
            attn_input_interv -= intervention_dict[(layer, ah_idx)]

        attn_input_interv_map[(layer, ah_idx)] = attn_input_interv
        
        # Computing the Q, K, V after the intervention
        # We use these values to do the actual intervention
        q_interv = F.linear(attn_input_interv, model.W_Q[layer, ah_idx, :, :].T, model.b_Q[layer, ah_idx, :])
        k_interv = F.linear(attn_input_interv, model.W_K[layer, ah_idx, :, :].T, model.b_K[layer, ah_idx, :])
        v_interv = F.linear(attn_input_interv, model.W_V[layer, ah_idx, :, :].T, model.b_V[layer, ah_idx, :])

        hook_fn_q = partial(intervention_fn_b, ah_idx=ah_idx, interv_value=q_interv)
        hook_fn_k = partial(intervention_fn_b, ah_idx=ah_idx, interv_value=k_interv)
        hook_fn_v = partial(intervention_fn_b, ah_idx=ah_idx, interv_value=v_interv)

        # Adds a hook into global model state
        model.blocks[layer].attn.hook_q.add_hook(hook_fn_q)
        model.blocks[layer].attn.hook_k.add_hook(hook_fn_k)
        model.blocks[layer].attn.hook_v.add_hook(hook_fn_v)

    # Runs the model, temporarily adds caching hooks and then removes *all* hooks after running, including the ablation hook.
    interv_logits, interv_cache = model.run_with_cache(ioi_dataset.toks)
    
    # Remove the hook
    model.reset_hooks(including_permanent=True)

    return interv_logits, interv_cache, attn_input_interv_map

def run_intervention_A(model, ioi_dataset, intervention_dict, intervention_type, is_boosting):
    def intervention_fn(x, hook, is_boosting, interv_value):
        print(f"Intervention in {hook.name}")
        assert x.shape == interv_value.shape
        # Intervention is boosting, so we sum the intervention value
        if is_boosting:
            x += interv_value
        else:
            x -= interv_value
        return x
        
    print(f"Running intervention with intervention_type={intervention_type} with is_boosting={is_boosting}")
    
    # Reset hooks
    model.reset_hooks(including_permanent=True)

    for layer in intervention_dict:
        print(f"Intervening in layer={layer}")

        hook_fn = partial(intervention_fn, is_boosting=is_boosting, interv_value=intervention_dict[layer])
        # Adds a hook into global model state
        model.blocks[layer].hook_attn_out.add_hook(hook_fn)
    
    # Runs the model, temporarily adds caching hooks and then removes *all* hooks after running, including the ablation hook.
    interv_logits, interv_cache = model.run_with_cache(ioi_dataset.toks)
    
    # Remove the hook
    model.reset_hooks(including_permanent=True)

    return interv_logits, interv_cache

def compute_logit_diff(ioi_dataset, logits, logits_interv):
    n_prompts = logits.size(0)  # Number of prompts
    
    # # Getting the logit of the IO token (END position) for each prompt
    last_token_logits_io = deepcopy(logits[torch.arange(n_prompts), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs])  # Shape [n_vocab]
    last_token_logits_s = deepcopy(logits[torch.arange(n_prompts), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs])  # Shape [n_vocab]
    logit_diff = last_token_logits_io - last_token_logits_s

    last_token_logits_io_interv = deepcopy(logits_interv[torch.arange(n_prompts), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs])  # Shape [n_vocab]
    last_token_logits_s_interv = deepcopy(logits_interv[torch.arange(n_prompts), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs])  # Shape [n_vocab]
    logit_diff_interv = last_token_logits_io_interv - last_token_logits_s_interv

    return logit_diff, logit_diff_interv

def run_intervention(U, VT, model, cache, logits, ioi_dataset, edges_ablation, svs_used, sv_signs, intervention_type, is_boosting, is_random):
    print("Running random intervention" if is_random else f"Running SVs intervention")

    if intervention_type not in ["A", "B"]:
        raise Exception("ablation_type must be either A or B")
    
    # Now, we need a dict of interventions, since each edge can have distinct AHs
    intervention_dict = {}

    intervention_pos_dict = {}
    
    # Set of prompts that can be ablated (with SVs)
    prompts_ablation = set()

    for edge_ablation in tqdm(edges_ablation, desc="Intervening for each edge"):
        if edge_ablation["type"] not in ["d", "s"]:
            raise Exception("edge_ablation['type'] must be either d or s")
    
        # Getting the layer and head indexes (src and dest nodes)
        layer_upstream, ah_idx_upstream = edge_ablation["upstream_node"]
        layer_downstream, ah_idx_downstream = edge_ablation["downstream_node"]

        # Getting the src and dest tokens
        dest_token_pos_list = ioi_dataset.word_idx[edge_ablation["dest_token"]] # Long[Tensor, 'n_prompts']
        src_token_pos_list = ioi_dataset.word_idx[edge_ablation["src_token"]] # Long[Tensor, 'n_prompts']

        assert len(dest_token_pos_list) == len(src_token_pos_list) == len(ioi_dataset)

        # We intervene in the dest_token if the edge is of type "d", otherwise we intervene in the src_token
        interv_pos_list = deepcopy(dest_token_pos_list) if edge_ablation["type"] == "d" else deepcopy(src_token_pos_list)

        assert len(interv_pos_list) == len(ioi_dataset)

        # Getting the intervention type
        projection_type = "U" if edge_ablation["type"] == "d" else "V"

        # Initializing the delta intervention tensor
        # This tensor will always have the same shape as the attn_out tensor (n_prompts, n_tokens, d_model)
        delta_interv_tensor = torch.zeros(cache[f"blocks.{layer_upstream}.hook_attn_out"].shape)
    
        for prompt_idx in range(len(ioi_dataset)):
            SVs_top = []

            # Getting the src and dest tokens for the prompt in this edge
            dest_token = dest_token_pos_list[prompt_idx].item()
            src_token = src_token_pos_list[prompt_idx].item()

            # Getting the position to intervene for the prompt in this edge
            pos_interv = interv_pos_list[prompt_idx].item()

            # Step 1: getting the SVs
            try:
                SVs_top = svs_used[prompt_idx, layer_downstream, ah_idx_downstream, dest_token, src_token]
                SVs_signs = torch.tensor(sv_signs[prompt_idx, layer_downstream, ah_idx_downstream, dest_token, src_token])
            except KeyError:
                SVs_top = [] # No SVs to use
            
            # Step 2: computing the projections
            if len(SVs_top) == 0:
                # No ablation possible
                continue 
            else:
                prompts_ablation.add(prompt_idx)
                P_u, P_v, P_u_perp, P_v_perp  = compute_projections(U[(layer_downstream, ah_idx_downstream)], VT[(layer_downstream, ah_idx_downstream)], SVs_top)
                P_u_rand, P_v_rand, P_u_rand_perp, P_v_rand_perp = compute_projections_random(U[(layer_downstream, ah_idx_downstream)], VT[(layer_downstream, ah_idx_downstream)], SVs_top, random)

            # Step 3: computing the delta used to intervene
            attn_out = cache[f"blocks.{layer_upstream}.attn.hook_result"][prompt_idx, :, ah_idx_upstream, :] # Float[Tensor, 'n_tokens d_model']

            # Step 3.1) computing the intervention value
            if is_random: # Random projection intervention
                P = P_u_rand if projection_type == "U" else P_v_rand
            else:
                P = P_u if projection_type == "U" else P_v
            attn_out_interv = projection_intervention(attn_out, P, pos_interv, projection_type) # Float[Tensor, 'n_tokens d_model']

            # Step 3.2) centering the intervention value (only in the position pos_interv)
            attn_out_interv[pos_interv] -= attn_out_interv[pos_interv].mean()
            assert torch.allclose(attn_out_interv[pos_interv].mean(), torch.zeros(1))
            if intervention_type == "B":
                # Step 3.3) scaling the intervention value (only for intervention type B)
                scaling_pos_interv = cache[f"blocks.{layer_downstream}.ln1.hook_scale"][prompt_idx, pos_interv] # Float[Tensor, 1]
                attn_out_interv[pos_interv] /= scaling_pos_interv
            
            # Step 3.4) putting the intervention value in the delta tensor (only in the position pos_interv)
            delta_interv_tensor[prompt_idx, pos_interv, :] += attn_out_interv[pos_interv]

            # Saving the position intervened for this prompt
            if intervention_type == "A":
                try:
                    intervention_pos_dict[layer_upstream].add((prompt_idx, pos_interv))
                except KeyError:
                    intervention_pos_dict[layer_upstream] = set([(prompt_idx, pos_interv)])
            elif intervention_type == "B":
                try:
                    intervention_pos_dict[(layer_downstream, ah_idx_downstream)].add((prompt_idx, pos_interv))
                except KeyError:
                    intervention_pos_dict[(layer_downstream, ah_idx_downstream)] = set([(prompt_idx, pos_interv)])

            # 1) attn_out and attn_out_interv can be different only in the position pos_interv
            assert torch.allclose(attn_out[:pos_interv], attn_out_interv[:pos_interv])
            assert torch.allclose(attn_out[pos_interv+1:], attn_out_interv[pos_interv+1:])
            assert not torch.allclose(attn_out[pos_interv], attn_out_interv[pos_interv])

            # 2) delta_interv_tensor can be non-zero only in the position pos_interv
            assert torch.allclose(delta_interv_tensor[prompt_idx, :pos_interv, :], torch.zeros((pos_interv, delta_interv_tensor.shape[2])))
            assert torch.allclose(delta_interv_tensor[prompt_idx, pos_interv+1:, :], torch.zeros((delta_interv_tensor.shape[1] - pos_interv - 1, delta_interv_tensor.shape[2])))
            assert not torch.allclose(delta_interv_tensor[prompt_idx, pos_interv, :], torch.zeros(delta_interv_tensor.shape[2]))

        # Adding the intervention tensor to the dict
        # Key: where we will intervene (layer) for type A and (layer, ah_idx) for type B
        # Value: the intervention (delta_interv_tensor)
        # If the key is in dict, we sum the intervention
        if intervention_type == "A":
            if layer_upstream in intervention_dict:
                print(f"Summing intervention in layer={layer_upstream}")
                intervention_dict[layer_upstream] += delta_interv_tensor
            else:
                intervention_dict[layer_upstream] = delta_interv_tensor
        elif intervention_type == "B":
            if (layer_downstream, ah_idx_downstream) in intervention_dict:
                print(f"Summing intervention in layer={layer_downstream}, ah_idx={ah_idx_downstream}")
                intervention_dict[(layer_downstream, ah_idx_downstream)] += delta_interv_tensor
            else:
                intervention_dict[(layer_downstream, ah_idx_downstream)] = delta_interv_tensor

    # Step 4: intervening in the model
    if intervention_type == "A":
        # Upstream node
        interv_logits, interv_cache = run_intervention_A(model, ioi_dataset, intervention_dict, intervention_type, is_boosting)
    elif intervention_type == "B":
        # Downstream node
        interv_logits, interv_cache, attn_input_interv_map = run_intervention_B(model, cache, ioi_dataset, intervention_dict, intervention_type, is_boosting)
    
    # Step 5: computing the ablation effect
    logit_diff, logit_diff_interv = compute_logit_diff(ioi_dataset, logits, interv_logits)

    # Step 6: computing how much we are changing in the intervention
    if intervention_type == "A":
        cosine_similarities = [[] for _ in range(len(ioi_dataset))]
        norm_ratio = [[] for _ in range(len(ioi_dataset))]
        for layer_upstream in intervention_dict.keys():
            list_interventions = list(intervention_pos_dict[layer_upstream])
            for prompt_id, interv_pos in list_interventions:
                before_interv = cache[f"blocks.{layer_upstream}.hook_attn_out"][prompt_id, interv_pos, :]
                after_interv = interv_cache[f"blocks.{layer_upstream}.hook_attn_out"][prompt_id, interv_pos, :]
                # Compute cosine similarity
                cos_sim = torch.cosine_similarity(before_interv, after_interv, dim=0).item()
                cosine_similarities[prompt_id].append(cos_sim)
                # Compute norm ratio
                norm_ratio[prompt_id].append(torch.norm(after_interv, dim=0).item() / torch.norm(before_interv, dim=0).item())
    elif intervention_type == "B":
        cosine_similarities = [[] for _ in range(len(ioi_dataset))]
        norm_ratio = [[] for _ in range(len(ioi_dataset))]
        for layer_downstream, ah_idx_downstream in intervention_pos_dict.keys():
            list_interventions = list(intervention_pos_dict[(layer_downstream, ah_idx_downstream)])
            for prompt_id, interv_pos in list_interventions:
                before_interv = cache[f"blocks.{layer_downstream}.ln1.hook_normalized"][prompt_id, interv_pos, :] # Input of the downstream AH.
                after_interv = attn_input_interv_map[(layer_downstream, ah_idx_downstream)][prompt_id, interv_pos, :] # Intervention in the downstream AH.
                assert before_interv.shape == after_interv.shape
                #after_interv = interv_cache[f"blocks.{layer_downstream}.ln1.hook_normalized"][prompt_id, interv_pos, :] # Input of the downstream AH.
                # Compute cosine similarity
                cos_sim = torch.cosine_similarity(before_interv, after_interv, dim=0).item()
                cosine_similarities[prompt_id].append(cos_sim)
                # Compute norm ratio
                norm_ratio[prompt_id].append(torch.norm(after_interv, dim=0).item() / torch.norm(before_interv, dim=0).item())

    # Step 6.1: taking the average of the cosine similarities and norm ratios per prompt
    cosine_similarities = np.array([np.mean(cos_sim) if len(cos_sim) > 0 else 1.0 for cos_sim in cosine_similarities])
    norm_ratio = np.array([np.mean(nr) if len(nr) > 0 else 1.0 for nr in norm_ratio])
    
    return logit_diff, logit_diff_interv, list(prompts_ablation), cosine_similarities, norm_ratio

def run_single_side_experiment(U, VT, G, EDGES_ABLATION, EDGES_NAMES=None):
    df_exp = pd.DataFrame()

    edges_ablation = []
    for edge in EDGES_ABLATION:
        edge_ablation = G.edges[edge]
        edge_ablation["upstream_node"] = eval(edge[0])
        edge_ablation["downstream_node"] = eval(edge[1])[:2]
        edges_ablation.append(edge_ablation)

    is_boosting, is_random = False, False

    if EDGES_NAMES is None:
        EDGES_NAMES = EDGES_ABLATION


    for intervention_type in ["A", "B"]: 
        intervention_type_name = intervention_type + " (Random)" if is_random else intervention_type + " (SVs)"

        print(f"Multi-edge intervention in {len(edges_ablation)} edges")
        # Multi-edge intervention
        logit_diff, logit_diff_interv, prompts_ablation, cosine_similarities, norm_ratio = run_intervention(U, VT, model, cache, logits, ioi_dataset, edges_ablation, svs_used, sv_signs, intervention_type, is_boosting, is_random)
        prompts_ablation_mask = [True if i in prompts_ablation else False for i in range(len(ioi_dataset))]

        df_exp = pd.concat([df_exp, pd.DataFrame({
            "prompt_id": range(len(ioi_dataset)),
            "edges": len(ioi_dataset) * [str(EDGES_NAMES)],
            "logit_diff": logit_diff,
            "logit_diff_interv": logit_diff_interv,
            "intervention_type": len(ioi_dataset) * [intervention_type],
            "intervention_type_name": len(ioi_dataset) * [intervention_type_name],
            "is_boosting": len(ioi_dataset) * [is_boosting],
            "is_random": len(ioi_dataset) * [is_random],
            "logit_diff_metric": logit_diff_interv - logit_diff,
            "is_ablated": prompts_ablation_mask,
            "cosine_similarities": cosine_similarities,
            "norm_ratio": norm_ratio
        })]).reset_index(drop=True)

        print("\n-----------------------------------\n")

    return df_exp

def run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B, EDGES_NAMES=None):
    df_exp = pd.DataFrame()

    edges_ablation_A = []
    for edge in EDGES_ABLATION_A:
        edge_ablation = G.edges[edge]
        edge_ablation["upstream_node"] = eval(edge[0])
        edge_ablation["downstream_node"] = eval(edge[1])[:2]
        edges_ablation_A.append(edge_ablation)

    edges_ablation_B = []
    for edge in EDGES_ABLATION_B:
        edge_ablation = G.edges[edge]
        edge_ablation["upstream_node"] = eval(edge[0])
        edge_ablation["downstream_node"] = eval(edge[1])[:2]
        edges_ablation_B.append(edge_ablation)

    if EDGES_NAMES is None:
        EDGES_NAMES = [EDGES_ABLATION_A, EDGES_ABLATION_B, EDGES_ABLATION_A + EDGES_ABLATION_B]

    is_boosting, is_random = False, False

    for idx, edges_ablation in enumerate([edges_ablation_A, edges_ablation_B, edges_ablation_A + edges_ablation_B]):
        for intervention_type in ["A", "B"]: 
            intervention_type_name = intervention_type + " (Random)" if is_random else intervention_type + " (SVs)"

            print(f"Multi-edge intervention in {len(edges_ablation)} edges")
            # Multi-edge intervention
            logit_diff, logit_diff_interv, prompts_ablation, cosine_similarities, norm_ratio = run_intervention(U, VT, model, cache, logits, ioi_dataset, edges_ablation, svs_used, sv_signs, intervention_type, is_boosting, is_random)
            prompts_ablation_mask = [True if i in prompts_ablation else False for i in range(len(ioi_dataset))]

            df_exp = pd.concat([df_exp, pd.DataFrame({
                "prompt_id": range(len(ioi_dataset)),
                "edges": len(ioi_dataset) * [str(EDGES_NAMES[idx])],
                "logit_diff": logit_diff,
                "logit_diff_interv": logit_diff_interv,
                "intervention_type": len(ioi_dataset) * [intervention_type],
                "intervention_type_name": len(ioi_dataset) * [intervention_type_name],
                "is_boosting": len(ioi_dataset) * [is_boosting],
                "is_random": len(ioi_dataset) * [is_random],
                "logit_diff_metric": logit_diff_interv - logit_diff,
                "is_ablated": prompts_ablation_mask,
                "cosine_similarities": cosine_similarities,
                "norm_ratio": norm_ratio
            })]).reset_index(drop=True)

            print("\n-----------------------------------\n")

    return df_exp

if __name__ == '__main__':
    # Load the model
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

    n_batches = 2
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=n_batches * 128,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=0,
        device=str("cpu")
    )

    # Load graph 
    G = nx.read_graphml("data/nms-p256-f1.0-folded-expandedO-scaled.graphml")

    # Load tracing data
    fname = f'data/results.nms-p256-f1.0-folded-expandedO-scaled.pkl'
    with open(fname, 'rb') as fp:
        (db, params) = pickle.load(fp)
    # Indexes: (prompt_id, layer, ah_idx, dest_token, src_token)
    (contrib_sv_src, contrib_sv_dest, sv_signs, sv_mags, svs_used) = db

    # Let's cache all the prompts
    logits, cache = model.run_with_cache(ioi_dataset.toks)

    # This creates the keys f"blocks.{layer}.attn.hook_result", which have the individual attention heads outputs.
    cache.compute_head_results()

    U, S, VT, _ = get_omega_decomposition_all_ahs(model)

    # Experiment A
    df_exp_A = run_two_sides_experiment(U, VT, G, [("(8, 6)", "(9, 9, 'end')")], [("(8, 6)", "(10, 0, 'end')")])

    # Experiment B
    EDGES_ABLATION_A = [("(7, 9)", "(8, 6, 'end')"), ("(7, 9)", "(9, 9, 'end')")]
    EDGES_ABLATION_B = [("(7, 3)", "(8, 6, 'end')"), ("(7, 3)", "(9, 9, 'end')")]
    df_exp_B = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B)

    # Experiment C
    EDGES_ABLATION_A = [("(7, 3)", "(9, 9, 'end')"), ("(7, 9)", "(9, 9, 'end')")]
    EDGES_ABLATION_B = [("(7, 3)", "(8, 6, 'end')"), ("(7, 9)", "(8, 6, 'end')")]
    df_exp_C = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B)

    # Experiment D
    EDGES_ABLATION_A = [("(0, 11)", "(7, 9, 'end')"), ("(1, 4)", "(7, 9, 'end')")]
    EDGES_ABLATION_B = [("(0, 9)", "(7, 9, 'S2')")]
    df_exp_D = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B)

    # Experiment E
    EDGES_ABLATION_A = list(G.in_edges("(9, 9, 'end')"))
    EDGES_ABLATION_B = list(G.in_edges("(9, 9, 'IO')"))
    EDGES_NAMES = ["All incoming edges to (9, 9, 'end')", "All incoming edges to (9, 9, 'IO')", "All incoming edges to (9, 9, 'end') + All incoming edges to (9, 9, 'IO')"]
    df_exp_E = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B, EDGES_NAMES)

    # Experiment F
    EDGES_ABLATION_A = list(G.in_edges("(9, 6, 'end')"))
    EDGES_ABLATION_B = list(G.in_edges("(9, 6, 'IO')"))
    EDGES_NAMES = ["All incoming edges to (9, 6, 'end')", "All incoming edges to (9, 6, 'IO')", "All incoming edges to (9, 6, 'end') + All incoming edges to (9, 6, 'IO')"]
    df_exp_F = run_two_sides_experiment(G, EDGES_ABLATION_A, EDGES_ABLATION_B, EDGES_NAMES)

    # Experiment G
    EDGES_ABLATION_A = list(G.in_edges("(10, 0, 'end')"))
    EDGES_ABLATION_B = list(G.in_edges("(10, 0, 'IO')"))
    EDGES_NAMES = ["All incoming edges to (10, 0, 'end')", "All incoming edges to (10, 0, 'IO')", "All incoming edges to (10, 0, 'end') + All incoming edges to (10, 0, 'IO')"]
    df_exp_G = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B, EDGES_NAMES)

    # Experiment H
    EDGES_ABLATION_A = list(G.in_edges("(8, 6, 'end')"))
    EDGES_ABLATION_B = list(G.in_edges("(8, 6, 'S2')"))
    EDGES_NAMES = ["All incoming edges to (8, 6, 'end')", "All incoming edges to (8, 6, 'S2')", "All incoming edges to (8, 6, 'end') + All incoming edges to (8, 6, 'S2')"]
    df_exp_H = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B, EDGES_NAMES)

    # Experiment I
    EDGES_ABLATION = list(G.out_edges("(2, 8)"))
    EDGE_NAME = "All outgoing edges from (2, 8)"
    df_exp_I = run_single_side_experiment(U, VT, G, EDGES_ABLATION, EDGE_NAME)

    # Experiment J
    EDGES_ABLATION_A = list(G.out_edges("(7, 9)"))
    EDGES_ABLATION_B = list(G.out_edges("(7, 3)"))
    EDGES_NAMES = ["All outgoing edges from (7, 9)", "All outgoing edges from (7, 3)", "All outgoing edges from (7, 9) + All outgoing edges from (7, 3)"]
    df_exp_J = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B, EDGES_NAMES)

    # Experiment K
    EDGES_ABLATION_A = list(G.out_edges("(0, 11)"))
    EDGES_ABLATION_B = list(G.out_edges("(0, 9)"))
    EDGES_NAMES = ["All outgoing edges from (0, 11)", "All outgoing edges from (0, 9)", "All outgoing edges from (0, 11) + All outgoing edges from (0, 9)"]
    df_exp_K = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B, EDGES_NAMES)

    # Experiment L
    EDGES_ABLATION_A = [("(0, 9)", "(7, 9, 'S2')")]
    EDGES_ABLATION_B = [("(7, 9)", "(9, 9, 'end')")]
    df_exp_L = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B)

    # Experiment M
    EDGES_ABLATION_A = [("(0, 9)", "(7, 9, 'S2')")]
    EDGES_ABLATION_B = [("(7, 9)", "(8, 6, 'end')")]
    df_exp_M = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B)

    # Experiment N
    EDGES_ABLATION_A = [("(0, 11)", "(7, 9, 'end')")]
    EDGES_ABLATION_B = [("(7, 9)", "(9, 9, 'end')")]
    df_exp_N = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B)

    # Experiment O
    EDGES_ABLATION_A = [("(0, 11)", "(7, 9, 'end')")]
    EDGES_ABLATION_B = [("(7, 9)", "(8, 6, 'end')")]
    df_exp_O = run_two_sides_experiment(U, VT, G, EDGES_ABLATION_A, EDGES_ABLATION_B)

    # Save the results
    df_exp_A["experiment"] = "A"
    df_exp_B["experiment"] = "B"
    df_exp_C["experiment"] = "C"
    df_exp_D["experiment"] = "D"
    df_exp_E["experiment"] = "E"
    df_exp_F["experiment"] = "F"
    df_exp_G["experiment"] = "G"
    df_exp_H["experiment"] = "H"
    df_exp_I["experiment"] = "I"
    df_exp_J["experiment"] = "J"
    df_exp_K["experiment"] = "K"
    df_exp_L["experiment"] = "L"
    df_exp_M["experiment"] = "M"
    df_exp_N["experiment"] = "N"
    df_exp_O["experiment"] = "O"

    # Concat all experiments
    df_exp_all = pd.concat([df_exp_A, df_exp_B, df_exp_C, df_exp_D, df_exp_E, df_exp_F, df_exp_G, df_exp_H, df_exp_I, df_exp_J, df_exp_K, df_exp_L, df_exp_M, df_exp_N, df_exp_O])
    df_exp_all.to_parquet("data/interventions_multi-edge.parquet")