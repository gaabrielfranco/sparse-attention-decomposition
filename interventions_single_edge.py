import torch
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

def run_intervention_B(model, cache, ioi_dataset, layer, ah_idx, interv_value, intervention_type, is_boosting):
    if intervention_type != "B":
        raise ValueError(f"Unknown intervention type {intervention_type}")
    
    # Run with hooks using the delta_86 to intervene
    def intervention_fn_b(x, hook, ah_idx, interv_value):
        print(f"Intervention B in {hook.name}")
        x[:, :, ah_idx, :] = interv_value
        return x
    
    print(f"Running intervention with intervention_type={intervention_type} with is_boosting={is_boosting}")
    
    # Getting the attention input (it's the same for all heads)
    attn_input = cache[f"blocks.{layer}.ln1.hook_normalized"]
    
    # Inverting the attention input
    attn_input_interv = deepcopy(attn_input)
    if is_boosting:
        attn_input_interv += interv_value
    else:
        attn_input_interv -= interv_value
        
    # Computing the Q, K, V after the intervention
    # We use these values to do the actual intervention
    q_interv = F.linear(attn_input_interv, model.W_Q[layer, ah_idx, :, :].T, model.b_Q[layer, ah_idx, :])
    k_interv = F.linear(attn_input_interv, model.W_K[layer, ah_idx, :, :].T, model.b_K[layer, ah_idx, :])
    v_interv = F.linear(attn_input_interv, model.W_V[layer, ah_idx, :, :].T, model.b_V[layer, ah_idx, :])

    # Reset hooks
    model.reset_hooks(including_permanent=True)

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

    return interv_logits, interv_cache, attn_input_interv

def run_intervention_A(model, ioi_dataset, layer, interv_value, intervention_type, is_boosting):
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

    hook_fn = partial(intervention_fn, is_boosting=is_boosting, interv_value=interv_value)

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

def run_intervention(U, VT, model, cache, logits, ioi_dataset, edge_ablation, svs_used, sv_signs, intervention_type, is_boosting, is_random):
    print("Running random intervention" if is_random else f"Running SVs intervention")

    if intervention_type not in ["A", "B"]:
        raise Exception("ablation_type must be either A or B")
    
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
    
    # List of prompts that can be ablated (with SVs)
    prompts_ablation = []
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
            prompts_ablation.append(prompt_idx)
            P_u, P_v, P_u_perp, P_v_perp = compute_projections(U[(layer_downstream, ah_idx_downstream)], VT[(layer_downstream, ah_idx_downstream)], SVs_top)
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
        delta_interv_tensor[prompt_idx, pos_interv, :] = attn_out_interv[pos_interv]

        # 1) attn_out and attn_out_interv can be different only in the position pos_interv
        assert torch.allclose(attn_out[:pos_interv], attn_out_interv[:pos_interv])
        assert torch.allclose(attn_out[pos_interv+1:], attn_out_interv[pos_interv+1:])
        assert not torch.allclose(attn_out[pos_interv], attn_out_interv[pos_interv])

        # 2) delta_interv_tensor can be non-zero only in the position pos_interv
        assert torch.allclose(delta_interv_tensor[prompt_idx, :pos_interv, :], torch.zeros((pos_interv, delta_interv_tensor.shape[2])))
        assert torch.allclose(delta_interv_tensor[prompt_idx, pos_interv+1:, :], torch.zeros((delta_interv_tensor.shape[1] - pos_interv - 1, delta_interv_tensor.shape[2])))
        assert not torch.allclose(delta_interv_tensor[prompt_idx, pos_interv, :], torch.zeros(delta_interv_tensor.shape[2]))
    
    # Step 4: intervening in the model
    if intervention_type == "A":
        # Upstream node
        interv_logits, interv_cache = run_intervention_A(model, ioi_dataset, layer_upstream, delta_interv_tensor, intervention_type, is_boosting)
    elif intervention_type == "B":
        # Downstream node
        interv_logits, interv_cache, after_interv = run_intervention_B(model, cache, ioi_dataset, layer_downstream, ah_idx_downstream, delta_interv_tensor, intervention_type, is_boosting)
    
    # Step 5: computing the ablation effect
    logit_diff, logit_diff_interv = compute_logit_diff(ioi_dataset, logits, interv_logits)

    # Computing the values before and after the intervention
    if intervention_type == "A":
        before_interv = cache[f"blocks.{layer_upstream}.hook_attn_out"] # Output of the upstream AH.
        after_interv = interv_cache[f"blocks.{layer_upstream}.hook_attn_out"]
    elif intervention_type == "B":
        before_interv = cache[f"blocks.{layer_downstream}.ln1.hook_normalized"] # Input of the downstream AH.

    # Step 6) computing how much we are changing in the intervention
    # Getting only the position that we intervened for each prompt. 
    after_interv_interv_pos = after_interv[range(len(ioi_dataset)), interv_pos_list, :] # Float[Tensor, 'n_prompts d_model']
    before_interv_interv_pos = before_interv[range(len(ioi_dataset)), interv_pos_list, :] # Float[Tensor, 'n_prompts d_model']

    norm_ratio = torch.norm(after_interv_interv_pos, dim=1) / torch.norm(before_interv_interv_pos, dim=1)

    # Step 6.1) compare how much we are changing using cosine similarity
    cos_sim = F.cosine_similarity(after_interv_interv_pos, before_interv_interv_pos, dim=1)

    # Step 7) computing the interv effect in the downstream head probability
    # The intervention effect is measured on the dest_token and src_token
    attn_scores_matrix = cache[f"blocks.{layer_downstream}.attn.hook_attn_scores"][:, ah_idx_downstream, :, :]
    scores_dest_src_downstream_ah = attn_scores_matrix[range(len(ioi_dataset)), dest_token_pos_list, src_token_pos_list]

    attn_scores_matrix_interv = interv_cache[f"blocks.{layer_downstream}.attn.hook_attn_scores"][:, ah_idx_downstream, :, :]
    scores_dest_src_downstream_ah_interv = attn_scores_matrix_interv[range(len(ioi_dataset)), dest_token_pos_list, src_token_pos_list]

    return logit_diff, logit_diff_interv, norm_ratio, cos_sim, prompts_ablation, scores_dest_src_downstream_ah, scores_dest_src_downstream_ah_interv

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

    logit_diff_ablations = pd.DataFrame()

    NODES = ["(9, 6, 'end')", "(9, 9, 'end')", "(10, 0, 'end')", "(9, 6, 'IO')", "(9, 9, 'IO')", "(10, 0, 'IO')", "(8, 6, 'S2')", "(8, 6, 'end')", "(0, 9)"]
        
    for node in tqdm(NODES):
        if node == "(0, 9)":
            EDGES_ABLATION = G.out_edges(node)
        else:
            EDGES_ABLATION = G.in_edges(node)
        print(f"\n\nRunning ablation for node {node} with {len(EDGES_ABLATION)} edges\n\n")
        for intervention_type in ["A", "B"]:
            for is_boosting in [False, True]:
                for is_random in [False, True]:
                    for edge in EDGES_ABLATION:
                        # Skip this edge (with src_token = 15)
                        if edge == ('(0, 9)', "(1, 0, 'end')"):
                            continue

                        print(f"Running ablation for edge {edge}")
                        edge_ablation = G.edges[edge]
                        edge_ablation["upstream_node"] = eval(edge[0])
                        edge_ablation["downstream_node"] = eval(edge[1])[:2]
                        logit_diff, logit_diff_interv, norm_ratio, cos_sim, prompts_ablation, scores_dest_src_downstream_ah, scores_dest_src_downstream_ah_interv = run_intervention(U, VT, model, cache, logits, ioi_dataset, edge_ablation, svs_used, sv_signs, intervention_type, is_boosting, is_random)
                        prompts_ablation_mask = [True if i in prompts_ablation else False for i in range(len(ioi_dataset))]

                        intervention_type_name = intervention_type + " (Random)" if is_random else intervention_type + " (SVs)"

                        # Saving the logit difference
                        logit_diff_ablations = pd.concat([logit_diff_ablations, pd.DataFrame({
                            "prompt_id": range(len(ioi_dataset)),
                            "downstream_node": len(ioi_dataset) * [node],
                            "edge": len(ioi_dataset) * [str(edge)],
                            "logit_diff": logit_diff,
                            "logit_diff_interv": logit_diff_interv,
                            "norm_ratio": norm_ratio,
                            "cosine_similarity": cos_sim,
                            "intervention_type": len(ioi_dataset) * [intervention_type],
                            "intervention_type_name": len(ioi_dataset) * [intervention_type_name],
                            "is_boosting": len(ioi_dataset) * [is_boosting],
                            "is_random": len(ioi_dataset) * [is_random],
                            "edge_weight": len(ioi_dataset) * [edge_ablation["weight"]],
                            "scores_dest_src_downstream_ah": scores_dest_src_downstream_ah,
                            "scores_dest_src_downstream_ah_interv": scores_dest_src_downstream_ah_interv,
                            "is_ablated": prompts_ablation_mask,
                            "scores_dest_src_diff_metric": scores_dest_src_downstream_ah_interv - scores_dest_src_downstream_ah,
                            "logit_diff_metric": logit_diff_interv - logit_diff
                        })]).reset_index(drop=True)
                        print(f"Done running ablation for edge {edge}")
                        print("\n-----------------------------------\n")

    logit_diff_ablations.to_parquet("data/interventions_single-edge.parquet", index=False)