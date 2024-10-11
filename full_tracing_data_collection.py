import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
from ioi_dataset import IOIDataset
from collections import namedtuple
import pickle
from utils import get_components_used, get_omega_decomposition_all_ahs

torch.set_grad_enabled(False)

DB = namedtuple('DB', 'contrib_sv_src contrib_sv_dest sv_signs sv_mags svs_used')
Params = namedtuple('Params', 'attn_thresh perc_contrib_thresh num_prompts')

def full_tracing_data_collection(use_svs=True):
    # Loading the model with no processing (fold_ln, etc.)
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

    # Setting up the dataset
    n_batches = 2
    N = 128 * n_batches

    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=0,
        device=str("cpu")
    )

    # Let's cache all the prompts
    _, cache = model.run_with_cache(ioi_dataset.toks)

    # Getting the omega decomposition
    U, S, VT, _ = get_omega_decomposition_all_ahs(model)

    # Now we can start the full tracing data collection
    attn_thresh = 0.5 # We only consider attention heads that fire on a given source/dest pair
    perc_contrib_thresh = 1.0
    num_prompts = N
    svs_used = {}
    sv_signs = {}
    sv_mags = {}
    contrib_sv_src = {}
    contrib_sv_dest = {}
    for prompt_id in tqdm(range(num_prompts)): 
        for layer in range(1, model.cfg.n_layers):
            X_batch = cache[f"blocks.{layer}.ln1.hook_normalized"]
            for ah_idx in range(model.cfg.n_heads):
                for dest_token in range(ioi_dataset.word_idx["end"][prompt_id]+1):
                    for src_token in range(1, dest_token+1):
                        # did the attention head fire on this source/dest combination?
                        if cache[f"blocks.{layer}.attn.hook_pattern"][prompt_id, ah_idx, dest_token, src_token].item() < attn_thresh:
                            continue
                        # Sometimes all attn_scores are negative; in this case, we don't consider this an "interesting" firing
                        # This generally happens for dest token early in the instance, where there are few
                        # preceding tokens 
                        if cache[f"blocks.{layer}.attn.hook_attn_scores"][prompt_id, ah_idx, dest_token, src_token].item() < 0:
                            continue
                        # if good firing, determine which SVs it used 
                        X = X_batch[prompt_id, :, :] #Float[Tensor, 'n_tokens d_model']
                        df = get_components_used(model, X, src_token, dest_token, layer, ah_idx, 
                                                U[(layer, ah_idx)], S[(layer, ah_idx)], VT[(layer, ah_idx)])
                        if use_svs:
                            last_sv_idx = np.where(df['sv_perc_contribution'].values > perc_contrib_thresh)[0][0]
                        else:
                            last_sv_idx = 63 # all SVs
                        svs = df.iloc[:last_sv_idx+1].idx.astype(int).values
                        # store the svs used
                        svs_used[prompt_id, layer, ah_idx, dest_token, src_token] = svs
                        # store the sign and magnitude (sqrt(singular_value)) of each SV used
                        # note that we know that the ah output (attn_score) is positive
                        sv_signs[prompt_id, layer, ah_idx, dest_token, src_token] = [np.sign(df.loc[df.idx == sv]['sim_i'].values[0]) for sv in svs]
                        sv_mags[prompt_id, layer, ah_idx, dest_token, src_token] = [np.sqrt(df.loc[df.idx == sv]['singular_value'].values[0]) for sv in svs]
                        VT_weighted = (torch.diag(torch.Tensor(sv_signs[prompt_id, layer, ah_idx, dest_token, src_token]))
                                        @ torch.diag(torch.Tensor(sv_mags[prompt_id, layer, ah_idx, dest_token, src_token]))
                                        @ VT[(layer, ah_idx)][svs])
                        U_weighted = (U[(layer, ah_idx)][:, svs]
                                        @ torch.diag(torch.Tensor(sv_signs[prompt_id, layer, ah_idx, dest_token, src_token]))
                                        @ torch.diag(torch.Tensor(sv_mags[prompt_id, layer, ah_idx, dest_token, src_token])))
                        contrib_src = np.zeros((layer, 12))
                        contrib_dest = np.zeros((layer, 12))
                        
                        # look at each upstream attention head
                        for prev_layer in range(layer):
                            for prev_ah_idx in range(12):
                                x_out = (
                                    cache[f"blocks.{prev_layer}.attn.hook_z"][:, :, prev_ah_idx, :] 
                                    @ model.W_O[prev_layer, prev_ah_idx, :, :])
                                contrib_src[prev_layer, prev_ah_idx] = (torch.einsum('ij,j->', VT_weighted, 
                                                                                    torch.cat([x_out[prompt_id, src_token, :], torch.ones(1)]))
                                                                                    / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, src_token, 0])
                                contrib_dest[prev_layer, prev_ah_idx] = (torch.einsum('ij,i->', U_weighted, 
                                                                                    torch.cat([x_out[prompt_id, dest_token, :], torch.ones(1)]))
                                                                                    / cache[f'blocks.{layer}.ln1.hook_scale'][prompt_id, dest_token, 0])
                        contrib_sv_src[prompt_id, layer, ah_idx, dest_token, src_token] = contrib_src
                        contrib_sv_dest[prompt_id, layer, ah_idx, dest_token, src_token] = contrib_dest

    # Save the results into a pickle file
    if use_svs:
        filename = f'data/results.nms-p{num_prompts}-f{perc_contrib_thresh}-folded-expandedO-scaled.pkl'
    else:
        filename = f'data/results.nms-p{num_prompts}-f{perc_contrib_thresh}-folded-expandedO-scaled-all-svs.pkl'
    with open(filename, 'wb') as fp:
        db = DB(contrib_sv_src, contrib_sv_dest, sv_signs, sv_mags, svs_used)
        params = Params(attn_thresh, perc_contrib_thresh, num_prompts)
        pickle.dump((db, params), fp)

if __name__ == "__main__":
    # print("Running full tracing data collection")
    # full_tracing_data_collection(use_svs=True)
    print("Running full tracing data collection with all SVs")
    full_tracing_data_collection(use_svs=False)
