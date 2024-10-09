from copy import deepcopy
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

def get_omega_decomposition_all_ahs(model):
    ALL_AHS = [(layer, ah_idx) for layer in range(model.cfg.n_layers) for ah_idx in range(model.cfg.n_heads)]

    U, S, VT, omega = {}, {}, {}, {}
    rank = model.W_Q.shape[-1]
    for (layer, ah_idx) in tqdm(ALL_AHS):
        omega[(layer, ah_idx)] = torch.column_stack([
                torch.row_stack([
                    model.W_Q[layer, ah_idx, :, :].cpu() @ model.W_K[layer, ah_idx, :, :].cpu().T, #Float[Tensor, 'd_model d_model']
                    model.b_Q[layer, ah_idx, :].cpu() @ model.W_K[layer, ah_idx, :, :].cpu().T
                ]),
                torch.row_stack([(model.W_Q[layer, ah_idx, :, :].cpu() @ model.b_K[layer, ah_idx, :]).reshape((-1,1)),
                    (model.b_Q[layer, ah_idx, :].cpu() @ model.b_K[layer, ah_idx, :]).reshape((1,1))
                ])
        ])
        Ubig, Sbig, VTbig = torch.linalg.svd(omega[(layer, ah_idx)])
        U[(layer, ah_idx)], S[(layer, ah_idx)], VT[(layer, ah_idx)] = Ubig[:,:rank], Sbig[:rank], VTbig[:rank]

    return U, S, VT, omega

def get_components_used(model, X, src_token, dest_token, layer, ah_idx, U, S, VT):
    # x_i is the destination token, x_j is the source token
    x_i = X[dest_token, :]
    x_i_tilde = torch.cat([x_i, torch.ones(1)])
    x_j = X[src_token, :]
    x_j_tilde = torch.cat([x_j, torch.ones(1)])

    # Computing the attention score directly
    attention_score = ((x_i @ model.W_Q[layer, ah_idx, :, :] @ model.W_K[layer, ah_idx, :, :].T @ x_j).item() 
                       + (model.b_Q[layer, ah_idx, :] @ model.W_K[layer, ah_idx, :, :].T @ x_j).item()
                       + (x_i @ model.W_Q[layer, ah_idx, :, :].cpu() @ model.b_K[layer, ah_idx, :]).item()
                       + (model.b_Q[layer, ah_idx, :].cpu() @ model.b_K[layer, ah_idx, :]).item())

    # Computing similarity
    sim = []
    k = 64
    for idx_k in range(k):
        u_k, v_k = U[:, idx_k], VT[idx_k, :]
        sim_i = x_i_tilde @ u_k
        sim_j = x_j_tilde @ v_k
        sim.append((idx_k, S[idx_k].item(), sim_i.item(), sim_j.item(), (S[idx_k]*sim_i*sim_j).item()))
        
    # Creating a dataframe for easier manipulation
    df = pd.DataFrame(sim, columns=["idx", "singular_value", "sim_i", "sim_j", "product"], dtype=np.float32)

    if np.abs(np.sum(df['product']) - attention_score) >= 0.001:
        print(f"Layer {layer}, AH {ah_idx}, Src Token {src_token}")
        print(f"Absolute error: {np.abs(np.sum(df['product']) - attention_score)}")

    # all three of these are equal - uncomment as a check
    # print(attention_score, x_i_tilde @ U @ torch.diag(S) @ VT @ x_j, df['product'].sum())
        
    # Cumsum of the products sorted in descending order divided by the first term of the attention
    # This gives the percentage of contribution of the singular values to the attention
    # We are interested in the SMALLEST number of singular values that contribute to 99% of the attention
    # Note that this value is always 1.0 when we use all singular values
    # sv_perc_contribution = np.cumsum(np.sort(df['product'])[::-1]) / first_attn_term
    if np.sum(df['product']) > 0:
        sv_perc_contribution = np.cumsum(np.sort(df['product'])[::-1]) / np.sum(df['product'])
        # Sort the dataframe in descending order
        df = df.sort_values(by="product", ascending=False)
    else:
        sv_perc_contribution = np.cumsum(np.sort(df['product'])) / np.sum(df['product'])
        # Sort the dataframe in ascending order
        df = df.sort_values(by="product", ascending=True)

    df["idx"] = df["idx"].astype(int)
    df["sv_perc_contribution"] = sv_perc_contribution

    return df

def get_ah_sv_set(ioi_dataset, model, cache, layer, ah_idx, U, S, VT, perc_contrib_thresh, thresh_firing):
    X_batch = cache[f"blocks.{layer}.ln1.hook_normalized"]
    sv_set = {"src_zero_firing": [], "non_firing": [], "non_src_zero_firing": []}
    contrib_set = {"src_zero_firing": [], "non_firing": [], "non_src_zero_firing": []}
    n_non_firing = 0
    n_non_src_zero_firing = 0
    n_src_zero_firing = 0
    sample_rate = 0.1
    N = X_batch.shape[0]
    for dest_token in range(X_batch.shape[1]): # Max number of tokens in the batch
        for src_token in range(1, dest_token+1): # skipping source zero; excluding it from analysis
            for prompt_id in range(N):
                X = X_batch[prompt_id, :, :] #Float[Tensor, 'n_tokens d_model']

                # We do not look to tokens after the end token
                if dest_token > ioi_dataset.word_idx["end"][prompt_id].item() or src_token > ioi_dataset.word_idx["end"][prompt_id].item():
                    continue
                
                # Sampling only {sample_rate} of the non-firing
                if cache[f"blocks.{layer}.attn.hook_pattern"][prompt_id, ah_idx, dest_token, src_token].item() < thresh_firing and np.random.rand() > sample_rate:
                    continue
                
                # Get the components used for this prompt
                df = get_components_used(model, X, src_token, dest_token, layer, ah_idx, U[(layer, ah_idx)], S[(layer, ah_idx)], VT[(layer, ah_idx)])
                
                last_sv_idx = np.where(df['sv_perc_contribution'].values > perc_contrib_thresh)[0][0]
                c = df.iloc[:last_sv_idx+1].idx.astype(int).values
                contrib = df.iloc[:last_sv_idx+1]['product'].values
                
                if cache[f"blocks.{layer}.attn.hook_pattern"][prompt_id, ah_idx, dest_token, src_token].item() < thresh_firing:
                    sv_set["non_firing"].append(c)
                    contrib_set["non_firing"].append(contrib)
                    n_non_firing += 1/sample_rate
                elif src_token == 0:
                    sv_set["src_zero_firing"].append(c)
                    contrib_set["src_zero_firing"].append(contrib)
                    n_src_zero_firing += 1
                else:
                    sv_set["non_src_zero_firing"].append(c)
                    contrib_set["non_src_zero_firing"].append(contrib)
                    n_non_src_zero_firing += 1
    
    return sv_set, contrib_set, n_non_firing, n_src_zero_firing, n_non_src_zero_firing

def get_heatmap_firings(ioi_dataset, model, cache, layer, ah_idx, U, S, VT, perc_contrib_thresh, thresh_firing):
    X_batch = cache[f"blocks.{layer}.ln1.hook_normalized"]
    N = X_batch.shape[0]
    heatmap_firings = {
        "all_firing": np.zeros((64, N), dtype=int), # 64 singular values, N prompts
        "non_src_zero_firing": np.zeros((64, N), dtype=int),
        "non_firing": np.zeros((64, N), dtype=int)
    }
    for dest_token in range(X_batch.shape[1]): # Max number of tokens in the batch
        for src_token in range(dest_token+1):
            for prompt_id in range(N):
                X = X_batch[prompt_id, :, :] #Float[Tensor, 'n_tokens d_model']

                # We do not look to tokens after the end token
                if dest_token > ioi_dataset.word_idx["end"][prompt_id].item() or src_token > ioi_dataset.word_idx["end"][prompt_id].item():
                    continue
                
                # Skipping non-firing
                if ((cache[f"blocks.{layer}.attn.hook_pattern"][prompt_id, ah_idx, dest_token, src_token].item() < thresh_firing)
                    and (np.random.rand() > 0.1)):
                    continue
                elif (cache[f"blocks.{layer}.attn.hook_pattern"][prompt_id, ah_idx, dest_token, src_token].item() < thresh_firing):
                    firing = False
                else:
                    firing = True
                
                # Get the components used for this prompt
                df = get_components_used(model, X, src_token, dest_token, layer, ah_idx, U[(layer, ah_idx)], S[(layer, ah_idx)], VT[(layer, ah_idx)])
                
                last_sv_idx = np.where(df['sv_perc_contribution'].values > perc_contrib_thresh)[0][0]
                c = df.iloc[:last_sv_idx+1].idx.astype(int).values

                if firing:
                    heatmap_firings["all_firing"][c, prompt_id] += 1
                    if src_token != 0:
                        heatmap_firings["non_src_zero_firing"][c, prompt_id] += 1
                else:
                    heatmap_firings["non_firing"][c, prompt_id] += 1
    
    return heatmap_firings


def get_contrib_src_dest(model, prompt_id, layer, ah_idx, src_token, dest_token, cache, U, S, VT, attn_thresh = 0.5, perc_contrib_thresh = 0.7):
    # return a matrix showing how much each upstream attention head's output would raise this ah's score
    X = cache[f"blocks.{layer}.ln1.hook_normalized"][prompt_id] #Float[Tensor, 'n_tokens d_model']
    # did the attention head fire on this source/dest combination?
    if cache[f"blocks.{layer}.attn.hook_pattern"][prompt_id, ah_idx, dest_token, src_token].item() < attn_thresh:
        print(f'No firing for {prompt_id, layer, ah_idx, dest_token, src_token}')
        return -1
    # Sometimes all attn_scores are negative; in this case, we don't consider this an "interesting" firing
    # This generally happens for dest token early in the instance, where there are few
    # preceding tokens 
    if cache[f"blocks.{layer}.attn.hook_attn_scores"][prompt_id, ah_idx, dest_token, src_token].item() < 0:
        print(f'Negative firing for {prompt_id, ah_idx, dest_token, src_token}')
        return -1
    if src_token == 0:
        print(f'Not considering src token 0: {prompt_id, ah_idx, dest_token, src_token}')
        return -1

    df = get_components_used(model, X, src_token, dest_token, layer, ah_idx, 
                                U[(layer, ah_idx)], S[(layer, ah_idx)], VT[(layer, ah_idx)])
    if perc_contrib_thresh == 'all':
        svs = range(64)
        sv_signs = np.array([1 for sv in svs])
    else:
        last_sv_idx = np.where(df['sv_perc_contribution'].values > perc_contrib_thresh)[0][0]
        svs = df.iloc[:last_sv_idx+1].idx.astype(int).values
        sv_signs = np.array([np.sign(df.loc[df.idx == sv]['sim_i'].values[0]) for sv in svs])


    sv_mags = np.array([np.sqrt(df.loc[df.idx == sv]['singular_value'].values[0]) for sv in svs])
    VT_local = np.diag(sv_signs) @ np.diag(sv_mags) @ (VT[(layer, ah_idx)][svs,:]).numpy()
    U_local = (U[(layer, ah_idx)][:, svs]).numpy() @ np.diag(sv_signs) @ np.diag(sv_mags)
    contrib_src  = np.zeros((layer, 12))
    contrib_dest = np.zeros((layer, 12))
    for prev_layer in range(layer):
        for prev_ah_idx in range(12):
            x_out = ((cache[f"blocks.{prev_layer}.attn.hook_z"][:, :, prev_ah_idx, :] 
                            @ model.W_O[prev_layer, prev_ah_idx, :, :])).numpy()
            contrib_src[prev_layer, prev_ah_idx] = (np.sum(VT_local 
                            @ np.append(x_out[prompt_id, src_token, :], 1))
                            / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, src_token, 0])
            contrib_dest[prev_layer, prev_ah_idx] = (np.sum(np.append(x_out[prompt_id, dest_token, :], 1) @ U_local)
                            / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, dest_token, 0])
    
    return contrib_src, contrib_dest, len(svs)

# Given the U and VT, compute the projections using the SVs in SVs_top
def compute_projections(U, VT, SVs_top):
    U_s = U[:, SVs_top]
    V_s = VT[SVs_top, :].T

    P_u = U_s @ U_s.T
    P_v = V_s @ V_s.T

    P_u_perp = torch.eye(P_u.shape[0]) - P_u
    P_v_perp = torch.eye(P_v.shape[0]) - P_v

    return P_u, P_v, P_u_perp, P_v_perp

def compute_projections_random(U, VT, SVs_top, random):
    # Random projection
    rank = U.shape[1]
    SVs_top_complement = [i for i in range(rank) if i not in SVs_top]

    # Sample random SVs from the complement with the same number of SVs
    SVs_rand = random.choice(SVs_top_complement, len(SVs_top), replace=False)

    # Compute the random projections
    U_s_rand = U[:, SVs_rand]
    V_s_rand = VT[SVs_rand, :].T

    P_u_rand = U_s_rand @ U_s_rand.T
    P_v_rand = V_s_rand @ V_s_rand.T

    P_u_rand_perp = torch.eye(P_u_rand.shape[0]) - P_u_rand
    P_v_rand_perp = torch.eye(P_v_rand.shape[0]) - P_v_rand

    return P_u_rand, P_v_rand, P_u_rand_perp, P_v_rand_perp

def projection_intervention(X, P, pos, intervention_type="U"):
    """
    X: the data to be intervened. Shape: (n_tokens, d_model)
    P: the projection matrix. Shape: either (d_model, d_model) or (d_model+1, d_model+1)
    pos: the position to be intervened (token index)
    intervention_type: "U" for U-projection, "V" for V-projection

    Returns the intervened data.
    """
    X_interv = deepcopy(X)
    if intervention_type == "U" or intervention_type == "V":
        # Adding the constant term in the position that we will intervene
        X_interv_pos = torch.cat([X_interv[pos],  torch.ones(1)])
        # We now apply the projection
        X_interv_pos = P @ X_interv_pos
        # Removing the constant term
        X_interv_pos = X_interv_pos[:-1]
        # Putting the intervened data back
        X_interv[pos] = X_interv_pos
    else:
        raise ValueError(f"Unknown intervention type {intervention_type}")
    
    return X_interv