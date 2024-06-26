# %%
from common.utils import *
pio.renderers.default = 'jupyterlab'
# %%
from transformer_lens import HookedTransformer
from datasets import load_dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model: HookedTransformer = HookedTransformer.from_pretrained("gelu-2l").to(device)
model.set_use_hook_mlp_in(True)
data = load_dataset("NeelNanda/c4-code-20k", split="train")
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
tokenized_data = tokenized_data.shuffle(seed=42)
all_tokens = tokenized_data["tokens"]
auto_encoder_run = "concat-z-gelu-21-l1-lr-sweep-3/gelu-2l_L1_Hcat_z_lr1.00e-03_l12.00e+00_ds16384_bs4096_dc1.00e-07_rie50000_nr4_v78"
encoder = AutoEncoder.load_from_hf(auto_encoder_run)
# %%
def get_encoder_hidden_acts(tokens, feature_id=None):
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
    
    _, _, hidden_acts, _, _ = encoder(acts)
    if feature_id is not None:
        hidden_acts = hidden_acts[:, feature_id]
        return hidden_acts.reshape(tokens.shape)
    return hidden_acts.reshape(tokens.shape + (-1,))

def get_token_with_space(token: int):
    str_tok = model.to_string(token)
    new_str_tok = " " + str_tok
    try:
        new_token = model.to_single_token(new_str_tok)
    except:
        return token
    return new_token

def remove_space_from_token(token: int):
    str_tok = model.to_string(token)
    if str_tok[0] != " ": return token
    new_str_tok = str_tok[1:]
    try:
        new_token = model.to_single_token(new_str_tok)
    except:
        return token
    return new_token

def get_uppercase_token(token: int):
    str_tok = model.to_string(token)
    new_str_tok = str_tok.capitalize()
    try:
        new_token = model.to_single_token(new_str_tok)
    except:
        return token
    return new_token

def get_lowercase_token(token: int):
    str_tok = model.to_string(token)
    new_str_tok = str_tok.lower()
    try:
        new_token = model.to_single_token(new_str_tok)
    except:
        return token
    return new_token

# %%
token_to_uppercase_token = {token: get_uppercase_token(token) for token in range(model.cfg.d_vocab)}
token_to_lowercase_token = {token: get_lowercase_token(token) for token in range(model.cfg.d_vocab)}
token_to_space_token = {token: get_token_with_space(token) for token in range(model.cfg.d_vocab)}
space_token_to_token = {token: remove_space_from_token(token) for token in range(model.cfg.d_vocab)}    
# %%
import collections
board_token = model.to_single_token('board')

def create_better_proxy_mask(tokens, board_token):
    batch_size, seq_len = tokens.size()
    mask = torch.zeros_like(tokens, dtype=torch.bool)

    for batch in range(batch_size):
        seen_tokens = collections.defaultdict(list)
        for pos in range(seq_len):
            token = tokens[batch, pos].item()
            space_token = token_to_space_token[token]
            nonspace_token = space_token_to_token[token]
            uppercase_token = token_to_uppercase_token[token]
            lowercase_token = token_to_lowercase_token[token]
            if token in seen_tokens and any(tokens[batch, past_pos+1] == board_token for past_pos in seen_tokens[token]):
                mask[batch, pos] = True
            elif space_token in seen_tokens and any(tokens[batch, past_pos+1] == board_token for past_pos in seen_tokens[space_token]):
                mask[batch, pos] = True
            elif nonspace_token in seen_tokens and any(tokens[batch, past_pos+1] == board_token for past_pos in seen_tokens[nonspace_token]):
                mask[batch, pos] = True
            elif uppercase_token in seen_tokens and any(tokens[batch, past_pos+1] == board_token for past_pos in seen_tokens[uppercase_token]):
                mask[batch, pos] = True
            elif lowercase_token in seen_tokens and any(tokens[batch, past_pos+1] == board_token for past_pos in seen_tokens[lowercase_token]):
                mask[batch, pos] = True
            seen_tokens[token].append(pos)

    return mask

# %%
## Specificity
data = []

# Loop over your batches to collect the activations
num_tokens = int(1e7)
feature_id = 14
batch_size = 256
num_batches = num_tokens // (batch_size * encoder.cfg["seq_len"])
for batch_idx in tqdm.trange(num_batches):
    tokens = all_tokens[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    hidden_acts = get_encoder_hidden_acts(tokens, feature_id=feature_id).cpu()

    better_proxy_mask = create_better_proxy_mask(tokens, board_token)

    act_mask = (hidden_acts > 0).cpu()

    hidden_act_values = torch.masked_select(hidden_acts, act_mask)
    is_better_proxy_active_values = torch.masked_select(better_proxy_mask, act_mask)

    act_indices = torch.nonzero(act_mask, as_tuple=False)

    act_indices[:, 0] += batch_idx * batch_size

    df_batch = pd.DataFrame({
        "hidden_act": utils.to_numpy(hidden_act_values),  # Convert to numpy array
        "is_better_proxy_active": utils.to_numpy(is_better_proxy_active_values),
        "all_tokens_batch_idx": utils.to_numpy(act_indices[:, 0]),
        "pos": utils.to_numpy(act_indices[:, 1])
    })

    data.append(df_batch)

df = pd.concat(data, ignore_index=True)
# %%
# Specificity plot
import plotly.express as px

fig = px.histogram(
    df, 
    x="hidden_act", 
    color="is_better_proxy_active", 
    barmode="stack", 
    nbins=100,
    labels={"is_better_proxy_active": "Is board induction"}  
)

fig.update_layout(
    xaxis_showgrid=True,
    yaxis_showgrid=True, 
    xaxis_showline=True, 
    yaxis_showline=True, 
    xaxis_linecolor='black', 
    yaxis_linecolor='black',
    plot_bgcolor='white',  
    xaxis_title='Feature Activation Level',
    yaxis_title='Count',
    title='Feature Activation Distribution',
)

fig.show()

# %%
#### Expected value plot
import plotly.graph_objects as go
import numpy as np
import pandas as pd

n_bins = 100

data_min = df['hidden_act'].min()
data_max = df['hidden_act'].max()
bin_edges = np.linspace(data_min, data_max, n_bins + 1)  # +1 because we need n_bins edges

fig = go.Figure()

for category in df['is_better_proxy_active'].unique():
    data = df[df['is_better_proxy_active'] == category]['hidden_act']
    counts, _ = np.histogram(data, bins=bin_edges, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected_values = bin_centers * counts / len(data)  # Normalize to get probability
    fig.add_trace(go.Bar(x=bin_centers, y=expected_values, name=str(category)))

fig.update_layout(
    barmode='stack',  # This makes the bars overlap
    plot_bgcolor='white',
    xaxis_title='Feature Activation Level',
    yaxis_title='Expected Value',
    title='Expected Value Plot',
)

fig.show()
# %%
# Look at dataset examples of false positives
def check_false_positives_in_range(num_examples_per_interval, lower, upper):
    print(f"Checking false positives in for acts in range: ({lower}, {upper})")
    false_positives_df = df.query("is_better_proxy_active==False & hidden_act > @lower & hidden_act < @upper")
    for b, pos in zip(false_positives_df["all_tokens_batch_idx"][:num_examples_per_interval], false_positives_df["pos"]):
        mask = torch.zeros_like(all_tokens[b], dtype=torch.float)
        mask[pos] = 1.
        create_html(model.to_str_tokens(all_tokens[b]), mask)

num_examples_per_interval = 3
check_false_positives_in_range(num_examples_per_interval, 4, 6)
# %%
# sensitivity section ##############################################################################
false_negatives = []

num_tokens = int(1e6)
num_batches = num_tokens // (batch_size * encoder.cfg["seq_len"])
for batch_idx in tqdm.trange(num_batches):
    tokens = all_tokens[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    hidden_acts = get_encoder_hidden_acts(tokens, feature_id=feature_id).cpu()
    better_proxy_mask = create_better_proxy_mask(tokens, board_token).cpu()
    fn_mask = (hidden_acts == 0) & better_proxy_mask
    fn_indices = torch.nonzero(fn_mask, as_tuple=False)
    fn_indices[:, 0] += batch_idx * batch_size
    false_negatives.extend([(idx.item(), pos.item()) for idx, pos in fn_indices])

print(len(false_negatives))

# %%
# Check false negatives 

num_examples = 3
for b, pos in false_negatives[:num_examples]:
    mask = torch.zeros_like(all_tokens[b], dtype=torch.float)
    mask[pos] = -1.
    hidden_acts = get_encoder_hidden_acts(all_tokens[b], feature_id=feature_id)
    hidden_acts[pos] = -hidden_acts.max()
    create_html(model.to_str_tokens(all_tokens[b]), hidden_acts)
# %%