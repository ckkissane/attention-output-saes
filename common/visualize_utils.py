# %%
from dataclasses import dataclass
from datasets import load_dataset
from eindex import eindex
import einops
from functools import partial
from jaxtyping import Float, Int
import os
from pathlib import Path
import time
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from typing import Dict, List, Optional, Tuple, Union
import IPython
from IPython.display import display, HTML
from huggingface_hub import hf_hub_download

from common.utils import DTYPES, device, get_batch_tokens, shuffle_data
from common.torch_modules.autoencoder import AutoEncoder

HTML_DIR = Path(__file__).parent.parent / "sae_visualizer/html"
HTML_HOVERTEXT_SCRIPT = (HTML_DIR / "hovertext_script.html").read_text()
HTML_TOKEN = """<span class="hover-text">
    <span class="token" style="background-color: bg_color; border-bottom: 4px solid underline_color; font-weight: font_weight">this_token</span>
    <div class="tooltip">
        <div class="table-container">
            <table>
                <tr><td class="right-aligned">Token</td><td class="left-aligned"><code>this_token</code></td></tr>
                <tr><td class="right-aligned">Feature activation</td><td class="left-aligned">feat_activation</td></tr>
                <tr><td colspan="2">resid_features_html</td></tr>
            </table>
            <!-- No effect! -->
          </div>
    </div>
</span>"""
CSS_DIR = Path(__file__).parent.parent / "sae_visualizer/css"

CSS = "\n".join([
    (CSS_DIR / "general.css").read_text(),
    (CSS_DIR / "sequences.css").read_text(),
    (CSS_DIR / "tables.css").read_text(),
])


def default_encoder(layer: int = 0) -> torch.nn.Module:
    assert 0 <= layer <= 11, "Layer must be b/w 0 to 11"

    encoders = {
        0: "gpt2-small_L0_Hcat_z_lr1.20e-03_l11.80e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        1: "gpt2-small_L1_Hcat_z_lr1.20e-03_l18.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v5",
        2: "gpt2-small_L2_Hcat_z_lr1.20e-03_l11.00e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v4",
        3: "gpt2-small_L3_Hcat_z_lr1.20e-03_l19.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        4: "gpt2-small_L4_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v7",
        5: "gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        6: "gpt2-small_L6_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        7: "gpt2-small_L7_Hcat_z_lr1.20e-03_l11.10e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        8: "gpt2-small_L8_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v6",
        9: "gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        10: "gpt2-small_L10_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v9",
        11: "gpt2-small_L11_Hcat_z_lr1.20e-03_l13.00e+00_ds24576_bs4096_dc3.16e-06_rsanthropic_rie25000_nr4_v9",
    }
    encoder = AutoEncoder.load_from_hf(encoders[layer], "ckkissane/attn-saes-gpt2-small-all-layers")
    return encoder

def default_model(encoder: Optional[torch.nn.Module] = None) -> torch.nn.Module:
    encoder = encoder or default_encoder(0)
    model = HookedTransformer.from_pretrained(encoder.cfg["model_name"]).to(DTYPES[encoder.cfg["enc_dtype"]]).to(encoder.cfg["device"])
    return model

def load_all_tokens(encoder: Optional[torch.nn.Module] = None, model: Optional[torch.nn.Module] = None, num_tokens: int = 1e7):
    if not encoder:
        encoder = default_encoder()
    
    if not model:
        model = default_model()

    DATA_DIR = Path("/workspace/data/")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
    dataset_iter = iter(dataset)

    all_tokens_batches = int(num_tokens) // encoder.cfg["seq_len"]

    try:
        print("Loading cached data from disk")
        all_tokens = torch.load("/workspace/data/owt_tokens_reshaped.pt")
        all_tokens = shuffle_data(all_tokens)
        print(all_tokens.shape)
    except:
        print("Data was not cached: Loading data first time")
        all_tokens = get_batch_tokens(dataset_iter, all_tokens_batches, model, encoder.cfg)
        torch.save(all_tokens, "/workspace/data/owt_tokens_reshaped.pt")
        print("all_tokens.shape", all_tokens.shape)
    
    return all_tokens

# %%
@dataclass
class MinimalSequenceData:
    '''
    Class to store data for a given sequence, which will be turned into a JavaScript visulisation.
    Basically just wraps token_ids and the corresponding feature acts for some (prompt, feature_id) pair

    Before hover:
        str_tokens: list of string tokens in the sequence
        feat_acts: sizes of activations on this sequence

    '''
    token_ids: List[str]
    feat_acts: List[float]

    def __len__(self):
        return len(self.token_ids)

    def __str__(self):
        return f"MinimalSequenceData({''.join(self.token_ids)})"

    def _filter(self, float_list: List[List[float]], int_list: List[List[str]]):
        float_list = [[f for f in floats if f != 0] for floats in float_list]
        int_list = [[i for i, f in zip(ints, floats)] for ints, floats in zip(int_list, float_list)]
        return float_list, int_list

class MinimalSequenceDataBatch:
    '''
    Class to store a list of MinimalSequenceData objects at once, by passing in tensors or objects
    with an extra dimension at the start.

    Note, I'll be creating these objects by passing in objects which are either 2D (k seq_len)
    or 3D (k seq_len top5), but which are all lists (of strings/ints/floats).

    '''
    def __init__(self, **kwargs):
        self.seqs = [
            MinimalSequenceData(
                token_ids = kwargs["token_ids"][k],
                feat_acts = kwargs["feat_acts"][k],
            )
            for k in range(len(kwargs["token_ids"]))
        ]

    def __getitem__(self, idx: int) -> MinimalSequenceData:
        return self.seqs[idx]

    def __len__(self) -> int:
        return len(self.seqs)

    def __str__(self) -> str:
        return "\n".join([str(seq) for seq in self.seqs])


def k_largest_indices(
    x: Float[Tensor, "rows cols"],
    k: int,
    largest: bool = True,
    buffer: Tuple[int, int] = (5, 5),
) -> Int[Tensor, "k 2"]:
    '''w
    Given a 2D array, returns the indices of the top or bottom `k` elements.

    Also has a `buffer` argument, which makes sure we don't pick too close to the left/right of sequence. If `buffer`
    is (5, 5), that means we shouldn't be allowed to pick the first or last 5 sequence positions, because we'll need
    to append them to the left/right of the sequence. We should only be allowed from [5:-5] in this case.
    '''
    x = x[:, buffer[0]:-buffer[1]]
    indices = x.flatten().topk(k=k, largest=largest).indices
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer[0]
    return torch.stack((rows, cols), dim=1)

def random_range_indices(
    x: Float[Tensor, "batch seq"],
    bounds: Tuple[float, float],
    k: int,
    buffer: Tuple[int, int] = (5, 5),
) -> Int[Tensor, "k 2"]:
    '''
    Given a 2D array, returns the indices of `k` elements whose values are in the range `bounds`.
    Will return fewer than `k` values if there aren't enough values in the range.

    Also has a `buffer` argument, which makes sure we don't pick too close to the left/right of sequence.
    '''
    # Limit x, because our indices (bolded words) shouldn't be too close to the left/right of sequence
    x = x[:, buffer[0]:-buffer[1]]

    # Creat a mask for where x is in range, and get the indices as a tensor of shape (k, 2)
    mask = (bounds[0] <= x) & (x <= bounds[1])
    indices = torch.stack(torch.where(mask), dim=-1)

    # If we have more indices than we need, randomly select k of them
    if len(indices) > k:
        indices = indices[sample_unique_indices(len(indices), k)]

    # Adjust indices to account for the buffer
    return indices + torch.tensor([0, buffer[0]]).to(indices.device)

def sample_unique_indices(large_number, small_number):
    '''Samples a small number of unique indices from a large number of indices.'''
    weights = torch.ones(large_number)  # Equal weights for all indices
    sampled_indices = torch.multinomial(weights, small_number, replacement=False)
    return sampled_indices

@torch.inference_mode()
def get_max_act_indices(
    encoder: AutoEncoder,
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    feature_idx: Union[int, List[int]],
    max_batch_size: Optional[int] = None,
    first_group_size: int = 20,
    verbose: bool = False,
):
    '''
    Gets (batch, pos) indices of the top activations for a given feature (or list of features)

    Args:
        feature_idx: int
            The identity of the feature we're looking at (i.e. we slice the weights of the encoder). A list of
            features is accepted (the result will be a list of FeatureData objects).
        max_batch_size: Optional[int]
            Optionally used to chunk the tokens, if it's a large batch

        buffer: Tuple[int, int]
            The number of tokens on either side of the feature, for the right-hand visualisation.

    Returns list of dictionaries that contain SequenceBatchData for each feature (see that class's docstring for more info).
    '''
    model.reset_hooks(including_permanent=True)

    # Make feature_idx a list, for convenience
    if isinstance(feature_idx, int): feature_idx = [feature_idx]
    n_feats = len(feature_idx)

    # Chunk the tokens, for less memory usage
    all_tokens = (tokens,) if max_batch_size is None else tokens.split(max_batch_size)
    all_tokens = [tok.to(device) for tok in all_tokens]

    # Create lists to store data, which we'll eventually concatenate
    all_feat_acts = []

    # Get encoder & decoder directions
    feature_act_dir = encoder.W_enc[:, feature_idx] # (d_mlp, feats)
    feature_bias = encoder.b_enc[feature_idx] # (feats,)
    #assert feature_act_dir.T.shape == feature_out_dir.shape #== (len(feature_idx), encoder.cfg.d_mlp)

    # ! Define hook function to perform feature ablation
    def hook_fn_act_post(act: Float[Tensor, "batch seq act_size"], hook: HookPoint):
        '''
        Encoder has learned x^j \approx b + \sum_i f_i(x^j)d_i where:
            - f_i are the feature activations
            - d_i are the feature output directions

        This hook function stores all the information we'll need later on. It doesn't actually perform feature ablation, because
        if we did this, then we'd have to run a different fwd pass for every feature, which is super wasteful! But later, we'll
        calculate the effect of feature ablation,  i.e. x^j <- x^j - f_i(x^j)d_i for i = feature_idx, only on the tokens we care
        about (the ones which will appear in the visualisation).
        '''
        if encoder.cfg["concat_heads"]:
            act = einops.rearrange(
                act, "batch seq n_heads d_head -> batch seq (n_heads d_head)",
            )
        x_cent = act - encoder.b_dec
        feat_acts_pre = einops.einsum(x_cent, feature_act_dir, "batch seq act_size, act_size feats -> batch seq feats")
        feat_acts = F.relu(feat_acts_pre + feature_bias)
        all_feat_acts.append(feat_acts)

    for _tokens in all_tokens:
        model.run_with_hooks(_tokens, return_type=None, fwd_hooks=[
            (encoder.cfg["act_name"], hook_fn_act_post),
        ])

    # Stack the results, and check shapes
    feat_acts = torch.concatenate(all_feat_acts) # [batch seq feats]
    assert feat_acts[:, :-1].shape == tokens[:, :-1].shape + (len(feature_idx),)

    iterator = range(n_feats) if not(verbose) else tqdm.tqdm(range(n_feats), desc="Getting sequence data", leave=False)

    indices_list = []
    for feat in iterator:

        _feat_acts = feat_acts[..., feat] # [batch seq]
        
        indices = k_largest_indices(_feat_acts, k=first_group_size, largest=True)
        indices_list.append(indices)
    
    return indices_list
        
@torch.inference_mode()
def get_seq_data(
    encoder: AutoEncoder,
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    feature_idx: Union[int, List[int]],
    max_batch_size: Optional[int] = None,

    buffer: Tuple[int, int] = (5, 5),
    n_groups: int = 10,
    first_group_size: int = 20,
    other_groups_size: int = 5,
    verbose: bool = False,

):
    '''
    Gets data that will be used to create the sequences in the HTML visualisation.

    Args:
        feature_idx: int
            The identity of the feature we're looking at (i.e. we slice the weights of the encoder). A list of
            features is accepted (the result will be a list of FeatureData objects).
        max_batch_size: Optional[int]
            Optionally used to chunk the tokens, if it's a large batch

        buffer: Tuple[int, int]
            The number of tokens on either side of the feature, for the right-hand visualisation.

    Returns list of dictionaries that contain SequenceBatchData for each feature (see that class's docstring for more info).
    '''
    model.reset_hooks(including_permanent=True)

    # Make feature_idx a list, for convenience
    if isinstance(feature_idx, int): feature_idx = [feature_idx]
    n_feats = len(feature_idx)

    # Chunk the tokens, for less memory usage
    all_tokens = (tokens,) if max_batch_size is None else tokens.split(max_batch_size)
    all_tokens = [tok.to(device) for tok in all_tokens]

    # Create lists to store data, which we'll eventually concatenate
    all_feat_acts = []

    # Get encoder & decoder directions
    feature_act_dir = encoder.W_enc[:, feature_idx] # (d_mlp, feats)
    feature_bias = encoder.b_enc[feature_idx] # (feats,)
    #assert feature_act_dir.T.shape == feature_out_dir.shape #== (len(feature_idx), encoder.cfg.d_mlp)

    # ! Define hook function to perform feature ablation
    def hook_fn_act_post(act: Float[Tensor, "batch seq act_size"], hook: HookPoint):
        '''
        Encoder has learned x^j \approx b + \sum_i f_i(x^j)d_i where:
            - f_i are the feature activations
            - d_i are the feature output directions

        This hook function stores all the information we'll need later on. It doesn't actually perform feature ablation, because
        if we did this, then we'd have to run a different fwd pass for every feature, which is super wasteful! But later, we'll
        calculate the effect of feature ablation,  i.e. x^j <- x^j - f_i(x^j)d_i for i = feature_idx, only on the tokens we care
        about (the ones which will appear in the visualisation).
        '''
        if encoder.cfg["concat_heads"]:
            act = einops.rearrange(
                act, "batch seq n_heads d_head -> batch seq (n_heads d_head)",
            )
        x_cent = act - encoder.b_dec
        feat_acts_pre = einops.einsum(x_cent, feature_act_dir, "batch seq act_size, act_size feats -> batch seq feats")
        feat_acts = F.relu(feat_acts_pre + feature_bias)
        all_feat_acts.append(feat_acts)

    # ! Run the forward passes (triggering the hooks), concat all results

    # Run the model without hook (to store all the information we need, not to actually return anything)
    for _tokens in all_tokens:
        model.run_with_hooks(_tokens, return_type=None, fwd_hooks=[
            (encoder.cfg["act_name"], hook_fn_act_post),
        ])

    # Stack the results, and check shapes
    feat_acts = torch.concatenate(all_feat_acts) # [batch seq feats]
    assert feat_acts[:, :-1].shape == tokens[:, :-1].shape + (len(feature_idx),)

    # ! Calculate all data for the right-hand visualisations, i.e. the sequences
    # TODO - parallelize this (it could probably be sped up by batching indices & doing all sequences at once, although those would be large tensors)
    # We do this in 2 steps:
    #   (1) get the indices per group, from the feature activations, for each of the 12 groups (top, bottom, 10 quantiles)
    #   (2) get a batch of SequenceData objects per group. This usually involves using eindex (i.e. indexing into the `tensors`
    #       tensor with the group indices), and it also requires us to calculate the effect of ablations (using feature activations
    #       and the clean residual stream values).

    sequence_data_list = []

    iterator = range(n_feats) if not(verbose) else tqdm.tqdm(range(n_feats), desc="Getting sequence data", leave=False)

    for feat in iterator:

        _feat_acts = feat_acts[..., feat] # [batch seq]
        
        k_largest_thing = k_largest_indices(_feat_acts, k=first_group_size, largest=True)

        # (1)
        indices_dict = {
            f"TOP ACTIVATIONS<br>MAX = {_feat_acts.max():.3f}": k_largest_indices(_feat_acts, k=first_group_size, largest=True),
            f"BOTTOM ACTIVATIONS<br>MIN = {_feat_acts.min():.3f}": k_largest_indices(_feat_acts, k=first_group_size, largest=False),
        }

        quantiles = torch.linspace(0, _feat_acts.max(), n_groups+1)
        for i in range(n_groups-1, -1, -1):
            lower, upper = quantiles[i:i+2]
            pct = ((_feat_acts >= lower) & (_feat_acts <= upper)).float().mean()
            indices = random_range_indices(_feat_acts, (lower, upper), k=other_groups_size)
            indices_dict[f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"] = indices

        # Concat all the indices together (in the next steps we do all groups at once)
        indices_full = torch.concat(list(indices_dict.values()))

        # (2)
        # ! We further split (2) up into 3 sections:
        #   (A) calculate the indices we'll use for this group (we need to get a buffer on either side of the target token for each seq),
        #       i.e. indices[..., 0] = shape (g, buf) contains the batch indices of the sequences, and indices[..., 1] = contains seq indices
        #   (B) index into all our tensors to get the relevant data (this includes calculating the effect of ablation)
        #   (C) construct the SequenceData objects, in the form of a SequenceDataBatch object

        # (A)
        # For each token index [batch, seq], we actually want [[batch, seq-buffer[0]], ..., [batch, seq], ..., [batch, seq+buffer[1]]]
        # We get one extra dimension at the start, because we need to see the effect on loss of the first token
        buffer_tensor = torch.arange(-buffer[0] - 1, buffer[1] + 1, device=indices_full.device)
        indices_full = einops.repeat(indices_full, "g two -> g buf two", buf=buffer[0] + buffer[1] + 2)
        indices_full = torch.stack([indices_full[..., 0], indices_full[..., 1] + buffer_tensor], dim=-1).cpu()

        # (B)
        # Template for indexing is new_tensor[k, seq] = tensor[indices_full[k, seq, 1], indices_full[k, seq, 2]], sometimes there's an extra dim at the end
        tokens_group = eindex(tokens, indices_full[:, 1:], "[g buf 0] [g buf 1]")
        feat_acts_group = eindex(_feat_acts, indices_full, "[g buf 0] [g buf 1]")

        # (C)
        # Now that we've indexed everything, construct the batch of SequenceData objects
        sequence_data = {}
        g_total = 0
        for group_name, indices in indices_dict.items():
            lower, upper = g_total, g_total + len(indices)
            sequence_data[group_name] = MinimalSequenceDataBatch(
                token_ids=tokens_group[lower: upper].tolist(),
                feat_acts=feat_acts_group[lower: upper, 1:].tolist(),
            )
            g_total += len(indices)

        # Add this feature's sequence data to the list
        sequence_data_list.append(sequence_data)

    return sequence_data_list

@dataclass 
class LogitData:
  top_negative_logits: List[Tuple[str, float]]
  top_positive_logits: List[Tuple[str, float]]
  
@dataclass
class DecoderWeightsDistribution:
  n_heads: int
  allocation_by_head: List[float]

from matplotlib import colors
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import re

BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list("bg_color_map", ["white", "darkorange"])

BLUE_BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list("bg_color_map", ["white", "#6495ED"])

def to_str_tokens_for_feature_card(vocab_dict: Dict[int, str], tokens: Union[int, torch.Tensor]):
    '''
    If tokens is 1D, does the same thing as model.to_str_tokens.
    If tokens is 2D or 3D, it flattens, does this thing, then reshapes.

    Also, makes sure that line breaks are replaced with their repr.
    '''
    if isinstance(tokens, int):
        return vocab_dict[tokens]

    assert tokens.ndim <= 3

    # Get flattened list of tokens
    str_tokens = [vocab_dict[t] for t in tokens.flatten().tolist()]

    # Replace line breaks with things that will appear as the literal '\n' in HTML
    str_tokens = [s.replace("\n", "&bsol;n") for s in str_tokens]
    # str_tokens = [s.replace(" ", "&nbsp;") for s in str_tokens]

    # Reshape
    return reshape(str_tokens, tokens.shape)

def generate_minimal_tok_html(
    vocab_dict: dict,
    this_token: str,
    bg_color: str,
    is_bold: bool = False,
    feat_act: float = 0.0,
    resid_features: Optional[Float[Tensor, "topk_ix 3"]] = None, # (value, feature, dfa)
):
    '''
    Creates a single sequence visualisation, by reading from the `token_template.html` file.

    Currently, a bunch of things are randomly chosen rather than actually calculated (we're going for
    proof of concept here).
    '''
    if resid_features is None:
        resid_features_html =  ""
    else:
        heaviest = resid_features[:, 2].argmax().item()
        def feat_link(feat_id, i):
            link = f"<a href='https://www.neuronpedia.org/gpt2-small/5-res-jb/{feat_id}'>Feat {feat_id}</a>"
            if heaviest == i:
                link = f"<b>{link}</b>"
            return link

        resid_features_html = "<div class='resid-features'>Top resid features: <ul>" + \
            "".join([f"<li>{feat_link(int(resid_features[i, 1]), i)} ({resid_features[i, 2]:0.2f}): Act {resid_features[i, 0]:0.2f}</li>" for i in range(resid_features.shape[0])]) + \
            f"</ul></div>"

    html_output = (
        HTML_TOKEN
        .replace("this_token", to_str_tokens_for_feature_card(vocab_dict, this_token))
        .replace("feat_activation", f"{feat_act:+.3f}")
        .replace("font_weight", "bold" if is_bold else "normal")
        .replace("bg_color", bg_color)
        .replace("resid_features_html", resid_features_html)
    )
    return html_output

def generate_minimal_seq_html(
    vocab_dict: dict,
    token_ids: List[str],
    feat_acts: List[float],
    resid_features: Optional[Float[Tensor, "example topk_index 3"]]=None,
    bold_idx: Optional[int] = None,
    color="orange"
):
    assert len(token_ids) == len(feat_acts), f"All input lists must be of the same length. token_ids: {len(token_ids)}, feat_acts: {len(feat_acts)}"

    # ! Clip values in [0, 1] range (temporary)
    #bg_values = np.clip(feat_acts, 0, 1)
    bg_values = feat_acts


    # Define the HTML object, which we'll iteratively add to
    html_output = '<div class="seq">'  # + repeat_obj

    for i in range(len(token_ids)):

        # Get background color, which is {0: transparent, +1: darkorange}
        bg_val = bg_values[i]
        bg_color = colors.rgb2hex(BG_COLOR_MAP(bg_val)) if color=="orange" else colors.rgb2hex(BLUE_BG_COLOR_MAP(bg_val))

        html_output += generate_minimal_tok_html(
            vocab_dict = vocab_dict,
            this_token = token_ids[i],
            bg_color = bg_color,
            is_bold = (bold_idx is not None) and (bold_idx == i),
            feat_act = feat_acts[i],
            resid_features = resid_features,
        )

    html_output += '</div>'
    return html_output

def get_per_src_pos_dfa_all_heads(tokens, feature_id, encoder, model):
    _, cache = model.run_with_cache(tokens)
    layer = encoder.cfg["layer"]
    v = cache["v", layer] # [batch, src_pos, n_heads, d_head]
    v_cat = einops.rearrange(v, "batch src_pos n_heads d_head -> batch src_pos (n_heads d_head)") # [batch, src_pos, n_heads*d_head]
    attn_weights = cache['pattern', layer]  # [batch, n_heads, dest_pos, src_pos]
    attn_weights_bcast = einops.repeat(attn_weights, "batch n_heads dest_pos src_pos -> batch dest_pos src_pos (n_heads d_head)", d_head=model.cfg.d_head)  # [batch, dest_pos, src_pos, n_heads*d_head]
    # Element-wise multiplication, removing the loop
    decomposed_z_cat = attn_weights_bcast * v_cat.unsqueeze(1)  # [batch, dest_pos, src_pos n_heads*d_head]
    per_src_pos_dfa = einops.einsum(
        decomposed_z_cat, encoder.W_enc[:, feature_id],
        "batch dest_pos src_pos d_model, d_model -> batch dest_pos src_pos",
    )
    return per_src_pos_dfa

@torch.inference_mode()
def get_resid_features(src_pos_indices: Int[Tensor, "src_pos"], max_act_tokens, feature_id, resid_encoder,
                       model, encoder, layer_ix, topk: int = 10) -> Float[Tensor, "example topk_index 2"]  :
    _, cache = model.run_with_cache(max_act_tokens)
    # Joseph's resid SAEs use `hook_point_layer` instead of `cfg["layer"]`
    layer = resid_encoder.cfg.hook_point_layer
    acts = cache["resid_pre", layer] # [batch, src_pos, d_model]
    _, hidden_acts, _, _, _, _ = resid_encoder(acts)
    assert tuple(hidden_acts.shape) ==  \
        tuple([max_act_tokens.shape[0],
                max_act_tokens.shape[1],
                resid_encoder.cfg.expansion_factor * resid_encoder.cfg.d_in])

    top_sae_features = torch.cat([
        (topk_obj := hidden_acts[torch.arange(src_pos_indices.shape[0]), src_pos_indices].topk(topk)).values.unsqueeze(-1),
        topk_obj.indices.unsqueeze(-1)
        ], dim=-1
    ) 

    # Apply DFA to top SAE features to determine which resid features are 
    # most used to compute z.
    # model.W_V [n_head d_model d_head]
    concat_WV = einops.rearrange(model.W_V[layer_ix], "n_head d_model d_head -> d_model (n_head d_head)") 

    v_dfa = (resid_encoder.W_dec[top_sae_features[:, :, 1].type(torch.int).reshape(-1),
                                :].reshape(top_sae_features.shape[0], top_sae_features.shape[1], -1) * \
        top_sae_features[:, :, 0].unsqueeze(-1) # Multiply out resid-decoder directions for the 20x10 features against resid-SAE activation values
    ) @ concat_WV @ encoder.W_enc[:, feature_id]

    top_sae_features = torch.cat([top_sae_features, v_dfa.unsqueeze(-1)], dim=-1)
    return top_sae_features # [example topk_index (value feature dfa)]

@torch.inference_mode()
def get_resid_sae_lens(feature_id, resid_encoder,
                       model, encoder, layer_ix, topk: int = 10) -> Float[Tensor, "topk_index 2"]  :
    if layer_ix == model.cfg.n_layers - 1: # There is currently no resid_post SAE
        return torch.zeros((topk, 2), dtype=torch.float, device=model.cfg.device)
    W_O_cat = einops.rearrange(
        model.W_O,
        "n_layers n_heads d_head d_model -> n_layers (n_heads d_head) d_model"
    )
    sae_logits = encoder.W_dec[feature_id] @ W_O_cat[layer_ix] @ resid_encoder.W_enc
    top_sae_logits = sae_logits.topk(topk)
    return torch.stack([top_sae_logits.values, top_sae_logits.indices], dim=-1) # [topk_index 2] each row is (score, resid_feature_id)

from functools import lru_cache

# should be able to pass feature_data[0] to this
@torch.inference_mode()
def get_minimal_sequences_html(sequence_data, vocab_dict, bold_idx,
                               logits_data: LogitData,
                               decoder_weights_distribution: DecoderWeightsDistribution,
                               tokens,
                               feature_id,
                               dfa_buffer,
                               max_act_indices,
                               layer_ix,
                               model,
                               encoder) -> str:

    sequences_html_dict = {}

    for group_name, sequences in sequence_data.items():

        full_html = f'<h4>{group_name}</h4>' # style="padding-left:25px;"

        for seq in sequences:
            html_output = generate_minimal_seq_html(
                vocab_dict,
                token_ids = seq.token_ids,
                feat_acts = seq.feat_acts,
                bold_idx = bold_idx, # e.g. the 6th item, with index 5, if buffer=(5, 5)
            )
            full_html += html_output

        sequences_html_dict[group_name] = full_html

    # Now, wrap all the values of this dictionary into grid-items: (top, groups of 3 for middle, bottom)
    html_top, html_bottom, *html_sampled = sequences_html_dict.values()
    sequences_html = ""
    sequences_html += f"<div class='grid-item'>{html_top}</div>"
    

    # Add per src DFA
    
    max_act_tokens = tokens[max_act_indices[:, 0].cpu()]
    per_src_dfa = get_per_src_pos_dfa_all_heads(max_act_tokens, feature_id, encoder, model) # [batch, dest_pos, src_pos]
    per_src_dfa = per_src_dfa[torch.arange(max_act_tokens.shape[0]), max_act_indices[:, 1], :] # [batch, src_pos]

    dfa_html = f'<h4>Top DFA by src position<br>MAX = {per_src_dfa.max():.3f}</h4>'


    # b = batch, pos = dest pos for max act example. 
    # max_dfa_idx will be the src pos 
    # To get the v-vector features, we will fetch them at the max_dfa_idx
    # [ix batch]
    src_acts_for_resid_sae = torch.empty((max_act_indices.shape[0], ), dtype=torch.int, device=tokens.device)

    dfa_token_ids_list = []
    per_src_dfa_sublist_lists = []
    for i, (b, pos) in enumerate(max_act_indices):
        b, pos = b.item(), pos.item()
        max_dfa_idx = per_src_dfa[i].argmax().item() 
        src_acts_for_resid_sae[i] = max_dfa_idx
        left_idx = max(0, max_dfa_idx - dfa_buffer[0])
        right_idx = min(len(per_src_dfa[i]), max_dfa_idx + dfa_buffer[1]+1)

        per_src_dfa_sublist_lists.append(per_src_dfa[i].tolist()[left_idx: right_idx])
        dfa_token_ids_list.append(max_act_tokens[i].tolist()[left_idx: right_idx])
    

    for i, (b, pos) in enumerate(max_act_indices):
        cur_html_output = generate_minimal_seq_html(
            vocab_dict,
            #token_ids = list(sequence_data.values())[0][i].token_ids,
            token_ids = dfa_token_ids_list[i],
            feat_acts = per_src_dfa_sublist_lists[i],
            resid_features = None,
            bold_idx = None, # e.g. the 6th item, with index 5, if buffer=(5, 5)
            color="blue"
        )
        dfa_html += cur_html_output
    
    sequences_html += f"<div class='grid-item'>{dfa_html}</div>"

    
    # Start of parent div to ensure vertical stacking
    sequences_html += "<div>"

    # First grid item: Decoder Weights Distribution
    sequences_html += "<div class='grid-item'><h4>Decoder Weights Distribution</h4>"
    for i in range(model.cfg.n_heads):
        sequences_html += f"<div style='display: inline-block; border-bottom: 1px solid #ccc'>Head {i}: {decoder_weights_distribution.allocation_by_head[i]:0.2f}</div><br />"
    sequences_html += "</div>"

    # Second grid item: Residual SAE Lens
    # next_residual_sae = load_residual_sae(layer_ix+1)
    # resid_sae_lens = get_resid_sae_lens(feature_id, next_residual_sae, model, encoder, layer_ix)
    # sequences_html += "<div class='grid-item'><h4>Residual SAE Lens</h4>"
    # for i in range(resid_sae_lens.shape[0]):
    #     # Update this line to include a clickable link
    #     cur_feature_id = int(resid_sae_lens[i, 1])
    #     sequences_html += f"<div style='display: inline-block; border-bottom: 1px solid #ccc'>"
    #     sequences_html += f"<a href='https://www.neuronpedia.org/gpt2-small/{layer_ix+1}-res-jb/{cur_feature_id}'>Feat {cur_feature_id}</a>: {resid_sae_lens[i, 0]:0.2f}</div><br />"
    # sequences_html += "</div>"

    # End of parent div
    sequences_html += "</div>"


    # First add top/bottom logits card
    opacity = lambda logit, logits: (logit - \
        (m := min(logit for _, logit in logits))) / \
        (max(logit for _, logit in logits) - m) / 4 + 0.5
    positive_opacity = partial(opacity, logits=logits_data.top_positive_logits)
    negative_opacity = partial(opacity, logits=logits_data.top_negative_logits)
    positive_logits_html = "".join(
      f"<div style='display: inline-block; border-bottom: 1px solid #ccc; width: 100%'><span style='background-color: rgba(171, 171, 255, {positive_opacity(logit)}); float: left;'>{token}</span><span style='float: right'>{logit:0.2f}</span></div><br />"
      for token, logit in logits_data.top_positive_logits
    )
    sequences_html += f"<div class='grid-item'><h4>Positive logits</h4>{positive_logits_html}</div>"
    negative_logits_html = "".join(
      f"<div style='display: inline-block; border-bottom: 1px solid #ccc; width: 100%'><span style='background-color: rgba(255, 149, 149, {negative_opacity(logit)}); float: left;'>{token}</span><span style='float: right'>{logit:0.2f}</span></div><br />"
      for token, logit in logits_data.top_negative_logits
    )
    sequences_html += f"<div class='grid-item'><h4>Negative logits</h4>{negative_logits_html}</div>"

    while len(html_sampled) > 0:
        L = min(3, len(html_sampled))
        html_next, html_sampled = html_sampled[:L], html_sampled[L:]
        sequences_html += "<div class='grid-item'>" + "".join(html_next) + "</div>"
    sequences_html += f"<div class='grid-item'>{html_bottom}</div>"

    return sequences_html + HTML_HOVERTEXT_SCRIPT

def style_minimal_sequences_html(sequences_html):
    return f"""
            <style>
            {CSS}
            </style>

            <div class='grid-container'>

                {sequences_html}

            </div>
    """

def get_vocab_dict(model):
    vocab_dict = model.tokenizer.vocab
    vocab_dict = {v: k.replace("Ġ", " ").replace("\n", "\\n") for k, v in vocab_dict.items()}
    return vocab_dict

def get_logit_data(encoder: AutoEncoder, model: nn.Module,
                   layer_ix: int, feature_idx: Union[int, List[int]],
                   top_k: int=20) -> List[LogitData]:
  if not isinstance(feature_idx, list):
    feature_idx = [feature_idx]

  logit_data = []
  flattened_WO = einops.rearrange(
    model.blocks[layer_ix].attn.W_O,
    "n_head d_head d_resid -> (n_head d_head) d_resid"
  )

  for feature in feature_idx:
    logits = encoder.W_dec[feature, :] @ flattened_WO @ model.W_U
    positive_logits = logits.topk(k=top_k, largest=True, sorted=True)
    negative_logits = logits.topk(k=top_k, largest=False, sorted=True)
    top_positive_logits = []
    top_negative_logits = []
    for token_ix, logit in zip(positive_logits.indices, positive_logits.values):
      top_positive_logits.append((model.tokenizer.decode(token_ix), float(logit)))
    for token_ix, logit in zip(negative_logits.indices, negative_logits.values):
      top_negative_logits.append((model.tokenizer.decode(token_ix), float(logit)))

    logit_data.append(LogitData(top_negative_logits, top_positive_logits))
    
  return logit_data

def get_decoder_weights_distribution(encoder: AutoEncoder, model: nn.Module,
    layer_ix: int, feature_idx: Union[int, List[int]]) -> List[DecoderWeightsDistribution]:
  if not isinstance(feature_idx, list):
    feature_idx = [feature_idx]

  distribs = []
  for feature in feature_idx:
    att_blocks = einops.rearrange(
        encoder.W_dec[feature, :], "(n_head d_head) -> n_head d_head", n_head=model.cfg.n_heads
    )
    decoder_weights_distribution = att_blocks.norm(dim=1) / att_blocks.norm(dim=1).sum()
    distribs.append(DecoderWeightsDistribution(model.cfg.n_heads, [float(x) for x in decoder_weights_distribution]))

  return distribs

### Creating visualizations

# TODO: Delete this.
def get_model_card(feature_idx: int, buffer=(10, 5)):
    
  feature_data = get_seq_data(
      encoder = encoder,
      model = model,
      tokens = all_tokens[:total_batch_size],
      feature_idx = feature_idx,
      max_batch_size = max_batch_size,
      buffer = buffer,
      n_groups = 10,
      first_group_size = 20,
      other_groups_size = 5,
      verbose = True,
  )

  logits_data: List[LogitData] = get_logit_data(
      encoder = encoder,
      model = model,
      layer_ix = encoder.cfg["layer"],
      feature_idx = feature_idx,
  )
  
  decoder_weights_distribution: List[DecoderWeightsDistribution] = \
  get_decoder_weights_distribution(
      encoder = encoder,
      model = model,
      layer_ix = encoder.cfg["layer"],
      feature_idx = feature_idx
  )

  vocab_dict = get_vocab_dict(model)
  sub_index = 0
  sequences_html = get_minimal_sequences_html(feature_data[sub_index], vocab_dict,
                                              bold_idx=buffer[0]+1, logits_data=logits_data[sub_index],
                                              decoder_weights_distribution=decoder_weights_distribution[sub_index],
                                              tokens=all_tokens[:total_batch_size],
                                              feature_id=feature_idx,
                                              dfa_buffer = buffer,
                                              model=model,
                                              encoder=encoder
                                              )
  html_string = style_minimal_sequences_html(sequences_html)
  display(HTML(html_string))

# %%
def get_feature_card_from_feature_data(feature_data, features, sub_index, layer_ix, buffer=(5, 5),
                                       max_batch_size=128, total_batch_size = None,
                                       display=True, max_act_indices=None, encoder=None, model=None, all_tokens=None):
    assert encoder is not None
    assert model is not None

    if all_tokens is None:
        if not "all_tokens" in globals():
            if hasattr(get_feature_card_from_feature_data, "all_tokens"):
                all_tokens = get_feature_card_from_feature_data.all_tokens
            else:
                all_tokens = get_feature_card_from_feature_data.all_tokens = load_all_tokens()
        else:
            all_tokens = globals()["all_tokens"]

    if total_batch_size is None:
        total_batch_size = int(1e6) // encoder.cfg["seq_len"],

    if max_act_indices is None:
        if hasattr(get_feature_card_from_feature_data, "max_act_indices") and get_feature_card_from_feature_data.last_features == features:
            max_act_indices = get_feature_card_from_feature_data.max_act_indices
        else:
            get_feature_card_from_feature_data.last_features = features
            max_act_indices = get_feature_card_from_feature_data.max_act_indices = \
                get_max_act_indices(
                    encoder,
                    model,
                    tokens=all_tokens[:total_batch_size], # Int[Tensor, "batch seq"]
                    feature_idx = features,
                    max_batch_size=max_batch_size,
                ) # TODO: maybe don't return list?

    feature_idx = features[sub_index]

    logits_data: List[LogitData] = get_logit_data(
        encoder = encoder,
        model = model,
        layer_ix = layer_ix,
        feature_idx = feature_idx,
    )
    
    decoder_weights_distribution: List[DecoderWeightsDistribution] = \
    get_decoder_weights_distribution(
        encoder = encoder,
        model = model,
        layer_ix = layer_ix,
        feature_idx = feature_idx
    )

    vocab_dict = get_vocab_dict(model)

    sequences_html = get_minimal_sequences_html(feature_data[sub_index], vocab_dict,
                                                bold_idx=buffer[0]+1, logits_data=logits_data[0],
                                                decoder_weights_distribution=decoder_weights_distribution[0],
                                                tokens=all_tokens[:total_batch_size],
                                                feature_id=feature_idx,
                                                dfa_buffer = (5, 5),
                                                max_act_indices=max_act_indices[sub_index],
                                                layer_ix=layer_ix,
                                                encoder=encoder,
                                                model=model)
    

    html_string = style_minimal_sequences_html(sequences_html)
    
    if display:
        IPython.display.display(HTML(html_string))
    else:
        return html_string

# %% 
def test_feature_card(layer_ix=0):
    if not "all_tokens" in globals():
        if hasattr(test_feature_card, "all_tokens"):
            all_tokens = test_feature_card.all_tokens
        else:
            all_tokens = load_all_tokens()

    features = [18, 17875]
    sub_idx = 1
    encoder = default_encoder(layer_ix)
    model = default_model(encoder)

    # this determines the chunks to break up batch into (for fwd passes)
    max_batch_size = 128
    # total number of datapoints (seq_len=128)
    total_batch_size = int(1e6) // encoder.cfg["seq_len"]

    all_feature_data = get_seq_data(
        encoder = encoder,
        model = model,
        tokens = all_tokens[:total_batch_size],
        feature_idx = features,
        max_batch_size = max_batch_size,
        buffer = (5, 5),
        n_groups = 10,
        first_group_size = 20,
        other_groups_size = 5,
        verbose = True,
    )

    t_max_indices_1 = time.time()
    max_act_indices = get_max_act_indices(
        encoder,
        model,
        tokens=all_tokens[:total_batch_size], # Int[Tensor, "batch seq"]
        feature_idx = features,
        max_batch_size=max_batch_size,
    ) # TODO: maybe don't return list?
    t_max_indices_2 = time.time()

    print(f"Getting max_act_indices took {t_max_indices_2-t_max_indices_1} seconds")

    feature_id = features[sub_idx]
    print("Feature ID:", feature_id)
    get_feature_card_from_feature_data(all_feature_data, features, sub_idx, layer_ix,
                                       max_batch_size=max_batch_size, total_batch_size=total_batch_size,
                                       max_act_indices=max_act_indices, model=model, encoder=encoder, all_tokens=all_tokens,
                                       display=True)

@lru_cache(maxsize=None)
def load_residual_sae(layer_ix: int) -> nn.Module:
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "mats_sae_training"))
    from sae_training.utils import LMSparseAutoencoderSessionloader

    REPO_ID = "jbloom/GPT2-Small-SAEs"
    FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer_ix}.hook_resid_pre_24576.pt"
    sae_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    _, sparse_autoencoder, activations_loader = LMSparseAutoencoderSessionloader.load_session_from_pretrained(sae_path)
    sparse_autoencoder.training = False
    return sparse_autoencoder