# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
os.environ["DATASETS_CACHE"] = "/workspace/cache/"
# %%
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio

# Janky code to do different setup when run in a Colab notebook vs VSCode
DEBUG_MODE = False
try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
    #%pip install git+https://github.com/neelnanda-io/TransformerLens.git
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    if ipython is not None:
        ipython.magic("load_ext autoreload")
        ipython.magic("autoreload 2")

if IN_COLAB:
    # Thanks to annoying rendering issues, Plotly graphics will either show up in colab OR Vscode depending on the renderer - this is bad for developing demos! Thus creating a debug mode.
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "png"

# Import stuff
import einops
import json
from pathlib import Path
import plotly.express as px
from torch.distributions.categorical import Categorical
import tqdm.notebook as tqdm
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from functools import partial

from IPython.display import HTML

import transformer_lens.utils as utils
import pandas as pd

from html import escape
import colorsys

from common.torch_modules.autoencoder import AutoEncoder

import wandb

import plotly.graph_objects as go

update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis",
     "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid",
     "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth"
}

def imshow(tensor, renderer=None, xaxis="", yaxis="", return_fig=False, **kwargs):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "facet_labels" in kwargs_pre:
        facet_labels = kwargs_pre.pop("facet_labels")
    else:
        facet_labels = None
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0,labels={"x":xaxis, "y":yaxis}, **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label

    if return_fig:
        return fig
    fig.show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(y=utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, return_fig=False, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    fig = px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)
    if return_fig:
        return fig
    fig.show(renderer)

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = utils.to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

def bar(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.bar(
        y=utils.to_numpy(tensor),
        labels={"x": xaxis, "y": yaxis},
        template="simple_white",
        **kwargs).show(renderer)

def create_html(strings, values, saturation=0.5, allow_different_length=False):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", "<br/>").replace("\t", "&emsp;").replace(" ", "&nbsp;")
        for s in escaped_strings
    ]

    if isinstance(values, torch.Tensor) and len(values.shape)>1:
        values = values.flatten().tolist()

    if not allow_different_length:
        assert len(processed_strings) == len(values)

    # scale values
    max_value = max(max(values), -min(values))+1e-3
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for i, s in enumerate(processed_strings):
        if i<len(scaled_values):
            v = scaled_values[i]
        else:
            v = 0
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(
            hue, v, 1
        )  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'

    display(HTML(html))

# %%
import argparse
def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg

# %%
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")
try:

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
except:
    pass


# %%
def shuffle_data(all_tokens):
    print("Shuffled data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]

# %%
class Buffer():
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty. 
    """
    def __init__(self, model, all_tokens, cfg):
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["act_size"]), dtype=torch.bfloat16, requires_grad=False).to(cfg["device"])
        self.cfg = cfg
        self.token_pointer = 0
        self.first = True
        self.all_tokens = all_tokens
        self.model = model
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.cfg["buffer_batches"]
            else:
                num_batches = self.cfg["buffer_batches"]//2
            self.first = False
            for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                tokens = self.all_tokens[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]
                _, cache = self.model.run_with_cache(tokens, stop_at_layer=self.cfg["layer"]+1, names_filter=self.cfg["act_name"])
                if self.cfg["concat_heads"]:
                    acts = einops.rearrange(
                        cache[self.cfg["act_name"]],
                        "batch pos head_index d_head -> (batch pos) (head_index d_head)",
                    )
                else:
                    acts = cache[self.cfg["act_name"]].reshape(-1, self.cfg["act_size"])
                
                #print(tokens.shape, acts.shape, self.pointer, self.token_pointer)
                self.buffer[self.pointer: self.pointer+acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]
                # if self.token_pointer > self.all_tokens.shape[0] - self.cfg["model_batch_size"]:
                #     self.token_pointer = 0

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0]//2 - self.cfg["batch_size"]:
            # print("Refreshing the buffer!")
            self.refresh()
        return out

# %%
class StreamingBuffer():
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty. 
    """
    def __init__(self, model, dataset_iter, cfg):
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["act_size"]), dtype=torch.bfloat16, requires_grad=False).to(cfg["device"])
        self.cfg = cfg
        self.token_pointer = 0
        self.first = True
        self.dataset_iter = dataset_iter
        self.model = model
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.cfg["buffer_batches"]
            else:
                num_batches = self.cfg["buffer_batches"]//2
            self.first = False
            for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                tokens = get_batch_tokens(self.dataset_iter, self.cfg["model_batch_size"], self.model, self.cfg)
                _, cache = self.model.run_with_cache(tokens, stop_at_layer=self.cfg["layer"]+1, names_filter=self.cfg["act_name"])
                if self.cfg["concat_heads"]:
                    acts = einops.rearrange(
                        cache[self.cfg["act_name"]],
                        "batch pos head_index d_head -> (batch pos) (head_index d_head)",
                    )
                else:
                    acts = cache[self.cfg["act_name"]].reshape(-1, self.cfg["act_size"])
                
                #print(tokens.shape, acts.shape, self.pointer, self.token_pointer)
                self.buffer[self.pointer: self.pointer+acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]
                # if self.token_pointer > self.all_tokens.shape[0] - self.cfg["model_batch_size"]:
                #     self.token_pointer = 0

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0]//2 - self.cfg["batch_size"]:
            # print("Refreshing the buffer!")
            self.refresh()
        return out

# %%
def get_batch_tokens(dataset_iter, batch_size, model, cfg):
    tokens = []
    total_tokens = 0
    while total_tokens < batch_size*cfg["seq_len"]:
        try:
            # Retrieve next item from iterator
            row = next(dataset_iter)["text"]
        except StopIteration:
            # Break the loop if dataset ends
            break
        
        # Tokenize the text with a check for max_length
        cur_toks = model.to_tokens(row)
        tokens.append(cur_toks)
        
        total_tokens += cur_toks.numel()

    # Check if any tokens were collected
    if not tokens:
        return None

    # Depending on your model's tokenization, you might need to pad the tokens here

    # Flatten the list of tokens
    flat_tokens = torch.cat(tokens, dim=-1).flatten()
    flat_tokens = flat_tokens[:batch_size * cfg["seq_len"]]
    reshaped_tokens = einops.rearrange(
        flat_tokens,
        "(batch seq_len) -> batch seq_len",
        batch=batch_size,
    )
    reshaped_tokens[:, 0] = model.tokenizer.bos_token_id
    return reshaped_tokens
# %%
def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

def replace_concat_z_hook(z, hook, encoder):
    z_cat = einops.rearrange(
        z,
        "batch pos head_index d_head -> (batch pos) (head_index d_head)",
    )
    z_reconstr_cat = encoder(z_cat)[1]
    z_reconstr = einops.rearrange(
        z_reconstr_cat,
        "(batch pos) (head_index d_head) -> batch pos head_index d_head",
        batch=z.shape[0], pos=z.shape[1], head_index=z.shape[2]
    )
    return z_reconstr

@torch.no_grad()
def get_recons_loss(model, all_tokens, num_batches, cfg, local_encoder=None, dataset_iter=None):
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    for i in range(num_batches):
        if dataset_iter is not None:
            tokens = get_batch_tokens(dataset_iter, cfg["model_batch_size"], model, cfg)
        else:
            tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        loss = model(tokens, return_type="loss")
        if cfg["concat_heads"]:
            recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], partial(replace_concat_z_hook, encoder=local_encoder))])
        else:
            recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], partial(replacement_hook, encoder=local_encoder))])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    
    ce_diff = recons_loss - loss
    return score, loss, recons_loss, zero_abl_loss, ce_diff

# %%
# Frequency
@torch.no_grad()
def get_freqs(model, all_tokens, num_batches, cfg, local_encoder=None, disable_tqdm=False, dataset_iter=None):
    if local_encoder is None:
        local_encoder = encoder
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).to(cfg["device"])
    total = 0
    for i in tqdm.trange(num_batches, disable=disable_tqdm):
        if dataset_iter is not None:
            tokens = get_batch_tokens(dataset_iter, cfg["model_batch_size"], model, cfg)
        else:
            tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        
        _, cache = model.run_with_cache(tokens, stop_at_layer=cfg["layer"]+1, names_filter=cfg["act_name"])
        acts = cache[cfg["act_name"]]
        if cfg["concat_heads"]:
            acts = einops.rearrange(
                acts,
                "batch pos head_index d_head -> (batch pos) (head_index d_head)",
            )
        else:
            acts = acts.reshape(-1, cfg["act_size"])

        hidden = local_encoder(acts)[2]
        
        #act_freq_scores += (hidden > 0).sum(0)
        act_freq_scores += torch.count_nonzero(hidden, dim=0)
        total+=hidden.shape[0]
        del tokens, cache, acts, hidden
    act_freq_scores /= total
    return act_freq_scores, total
# %%
@torch.no_grad()
def re_init(indices, encoder):
    if indices.dtype == torch.bool:
        indices = torch.arange(encoder.d_hidden, device=encoder.device)[indices]
    new_W_enc = torch.nn.init.kaiming_uniform_(
        torch.empty(
            encoder.cfg["act_size"], indices.shape[0], dtype=encoder.dtype, device=encoder.device
        )
    )
    new_W_enc /= torch.norm(new_W_enc, dim=0, keepdim=True)
    encoder.W_enc.data[:, indices] = new_W_enc
    if indices.shape[0] < encoder.d_hidden:
        sum_of_all_norms = torch.norm(encoder.W_enc.data, dim=0).sum()
        sum_of_all_norms -= len(indices)
        average_norm = sum_of_all_norms / (encoder.d_hidden - len(indices))
        # metrics["resample_norm_thats_hopefully_less_or_around_one"] = average_norm.item()
        encoder.W_enc.data[:, indices] *= encoder.cfg["resample_factor"] * average_norm
    else:
        # Whatever, norm 1 times resample factor seems fiiiiine
        encoder.W_enc.data[:, indices] *= encoder.cfg["resample_factor"]
    new_W_enc *= encoder.cfg["resample_factor"]
    new_b_enc = torch.zeros(
        indices.shape[0], dtype=encoder.dtype, device=encoder.device
    )
    encoder.b_enc.data[indices] = new_b_enc
    encoder.W_dec.data[indices, :] = encoder.W_enc.data[:, indices].T # Clone em!
    encoder.W_dec.data[indices, :] /= torch.norm(encoder.W_dec.data[indices, :], dim=1, keepdim=True)
    encoder.W_dec /= torch.norm(encoder.W_dec, dim=1, keepdim=True)

@torch.no_grad()
def anthropic_resample(buffer, indices, encoder, verbose=False):
    """
    Implements anthropic neuron resampling, as described in the paper, but with only one batch
    
    Args:
        acts: [batch_size, act_size]
        indices: [d_hidden] bool tensor, True if neuron is dead
        encoder: AutoEncoder
        
    Returns:
        None, but encoder and decoder weights are updated in place
    """
    assert indices.dtype == torch.bool
    alive_neurons = torch.arange(encoder.d_hidden, device=encoder.device)[~indices] # [n_alive,]
    dead_features = torch.arange(encoder.d_hidden, device=encoder.device)[indices] # [n_dead,]
        
    n_dead = dead_features.numel()
    
    # Get batch of input acts
    global_losses = torch.zeros((encoder.cfg["anthropic_resample_batches"]*encoder.cfg["batch_size"],), dtype=encoder.dtype, device=encoder.device)
    global_input_activations = torch.zeros((encoder.cfg["anthropic_resample_batches"]*encoder.cfg["batch_size"], encoder.cfg["act_size"]), dtype=encoder.dtype, device=encoder.device)
    pbar = tqdm.tqdm(range(encoder.cfg["anthropic_resample_batches"])) if verbose else range(encoder.cfg["anthropic_resample_batches"])
    for i in pbar:
        # compute loss for each element in batch
        acts = buffer.next() # [batch_size, act_size]
        loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts, per_token=True) # l2_loss: [batch_size]
        if l2_loss.max() < 1e-6:
            return # If we have zero reconstruction loss, we don't need to resample neurons
        global_losses[i*encoder.cfg["batch_size"]:(i+1)*encoder.cfg["batch_size"]] = l2_loss
        global_input_activations[i*encoder.cfg["batch_size"]:(i+1)*encoder.cfg["batch_size"]] = acts
    
    distn = Categorical(probs = global_losses.pow(2) / global_losses.pow(2).sum())
    #replacement_indices = distn.sample((n_dead,)) # shape [n_dead]
    replacement_indices = torch.multinomial(distn.probs, num_samples=n_dead, replacement=False) # sample without replacement ig?

    # Index into the batch of hidden activations to get our replacement values
    replacement_values = (global_input_activations - encoder.b_dec)[replacement_indices] # shape [n_dead n_input_ae]
    replacement_values_normalized = replacement_values / (replacement_values.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
    W_enc_norm_alive_mean = 1.0 if len(alive_neurons) == 0 else encoder.W_enc[:, alive_neurons].norm(dim=0).mean().item()

    # Lastly, set the new weights & biases
    # For W_dec (the dictionary vectors), we just use the normalized replacement values
    encoder.W_dec.data[dead_features, :] = replacement_values_normalized
    # For W_enc (the encoder vectors), we use the normalized replacement values scaled by (mean alive neuron norm * neuron resample scale)
    encoder.W_enc.data[:, dead_features] = replacement_values_normalized.T * W_enc_norm_alive_mean * encoder.cfg["anthropic_neuron_resample_scale"]
    # For b_enc (the encoder bias), we set it to zero
    encoder.b_enc.data[dead_features] = 0.0

# %%
def reset_optimizer(indices, opt, cfg):
    # Reset all the Adam parameters
    for dict_idx, (k, v) in enumerate(opt.state.items()):
        for v_key in ["exp_avg", "exp_avg_sq"]:
            if dict_idx == 0: # W_enc
                assert k.data.shape == (cfg["act_size"], cfg["dict_size"])
                v[v_key][:, indices] = 0.0
            elif dict_idx == 2: # b_enc
                assert k.data.shape == (cfg["dict_size"],)
                v[v_key][indices] = 0.0
            elif dict_idx == 1: # W_dec
                assert k.data.shape == (cfg["dict_size"], cfg["act_size"])
                v[v_key][indices, :] = 0.0
            elif dict_idx == 3: # b_dec
                assert k.data.shape == (cfg["act_size"],)
            else:
                raise ValueError(f"Unexpected dict_idx {dict_idx}")

    # Check that the opt is really updated
    for dict_idx, (k, v) in enumerate(opt.state.items()):
        for v_key in ["exp_avg", "exp_avg_sq"]:
            if dict_idx == 0:
                if k.data.shape != (cfg["act_size"], cfg["dict_size"]):
                    print(
                        "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                    )
                if v[v_key][:, indices].abs().max().item() > 1e-6:
                    print(
                        "Warning: it does not seem as if resetting the Adam parameters worked"
                    )

# %% 
@torch.no_grad()
def get_l0_norm(model, all_tokens, num_batches, cfg, local_encoder=None, disable_tqdm=False, dataset_iter=None):
    """
    Function to compute the L0 norm: aka the average number of nonzero entries in a sparse representation.
    Anthropic paper said they target a value less than 10 or 20, and distrust solutions where the L0 norm is a signifcant fraction of the transformer's activation dimensionality
    """
    if local_encoder is None:
        local_encoder = encoder
    avg_l0_norm = 0
    for _ in tqdm.trange(num_batches, disable=disable_tqdm):
        if dataset_iter is not None:
            tokens = get_batch_tokens(dataset_iter, cfg["model_batch_size"], model, cfg)
        else:
            tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        
        _, cache = model.run_with_cache(tokens, stop_at_layer=cfg["layer"]+1, names_filter=cfg["act_name"])
        acts = cache[cfg["act_name"]] # [batch, pos, act_size]
        acts = acts.reshape(-1, cfg["act_size"]) # [batch*pos, act_size]

        hidden = local_encoder(acts)[2] # [batch*pos, dict_size]

        #avg_l0_norm += (hidden > 0).sum(dim=-1).float().mean() # on a given token, how many features fired?
        avg_l0_norm += torch.count_nonzero(hidden, dim=-1).to(torch.bfloat16).mean()
        
    avg_l0_norm /= num_batches
    return avg_l0_norm

SPACE = "·"
NEWLINE="↩"
TAB = "→"

def process_token(s):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

def process_tokens(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [process_token(s) for s in l]

def process_tokens_index(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [f"{process_token(s)}/{i}" for i,s in enumerate(l)]

def create_vocab_df(logit_vec, model, make_probs=False, full_vocab=None):
    if full_vocab is None:
        full_vocab = process_tokens(model.to_str_tokens(torch.arange(model.cfg.d_vocab)))
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": utils.to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = utils.to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = utils.to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)

# %%
def get_encoder_hidden_acts(tokens, encoder, model, feature_id=None):
    _, cache = model.run_with_cache(
        tokens,
        return_type=None,
        names_filter = encoder.cfg["act_name"]
    )
    acts = cache[encoder.cfg["act_name"]]
    if encoder.cfg["concat_heads"]:
        acts = einops.rearrange(
            acts,
            "batch pos n_heads d_head -> (batch pos) (n_heads d_head)"
        )
    else:
        acts = acts.reshape(-1, encoder.cfg["act_size"])
    
    loss, x_reconstruct, hidden_acts, l2_loss, l1_loss = encoder(acts)
    if feature_id is not None:
        hidden_acts = hidden_acts[:, feature_id]
        return hidden_acts.reshape(tokens.shape)
    return hidden_acts.reshape(tokens.shape + (-1,))
# %%
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

# %%
def get_encoder_hidden_acts_from_acts(tokens, acts, encoder, feature_id=None):
    if encoder.cfg["concat_heads"]:
        acts = einops.rearrange(
            acts,
            "batch pos n_heads d_head -> (batch pos) (n_heads d_head)"
        )
    else:
        acts = acts.reshape(-1, encoder.cfg["act_size"])
    
    loss, x_reconstruct, hidden_acts, l2_loss, l1_loss = encoder(acts)
    if feature_id is not None:
        hidden_acts = hidden_acts[:, feature_id]
        return hidden_acts.reshape(tokens.shape)
    return hidden_acts.reshape(tokens.shape + (-1,))
# %%
def replace_with_recons_acts_hook(act, hook, encoder, ablated_feature=None, pinned_feature=None, pinned_value=None):
    # Decode activation into features
    if encoder.cfg["concat_heads"]:
        assert act.ndim == 4
        reshaped_acts = einops.rearrange(
            act,
            "batch pos n_heads d_head -> (batch pos) (n_heads d_head)"
        )
    else:
        reshaped_acts = act.reshape(-1, encoder.cfg["act_size"])
    loss, x_reconstruct, hidden_acts, l2_loss, l1_loss = encoder(reshaped_acts)
    if ablated_feature is not None:
        hidden_acts[..., ablated_feature] = 0
    if pinned_feature is not None:
        assert pinned_value is not None
        hidden_acts[..., pinned_feature] = pinned_value
    recons_acts = hidden_acts @ encoder.W_dec + encoder.b_dec
    recons_acts = recons_acts.reshape(act.shape)
    act[:] = recons_acts
    return act