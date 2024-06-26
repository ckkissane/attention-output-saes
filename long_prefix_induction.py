# %%
from common.utils import *
from transformer_lens import HookedTransformer
from datasets import load_dataset
import random
# %%
pio.renderers.default = 'jupyterlab'
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# %%
### Loading models and data
auto_encoder_run = "gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9"
encoder = AutoEncoder.load_from_hf(auto_encoder_run, hf_repo="ckkissane/attn-saes-gpt2-small-all-layers")
model = HookedTransformer.from_pretrained(encoder.cfg["model_name"]).to(DTYPES[encoder.cfg["enc_dtype"]]).to(encoder.cfg["device"])
# %%
DATA_DIR = Path("./data/")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
# %%
dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
dataset_iter = iter(dataset)

all_tokens_batches = int(1e7) // encoder.cfg["seq_len"]

try:
    print("Loading cached data from disk")
    all_tokens = torch.load(DATA_DIR / "owt_tokens_reshaped.pt")
    all_tokens = shuffle_data(all_tokens)
    print(f"all_tokens.shape: {all_tokens.shape}")
except:
    print("Data was not cached: Loading data first time")
    all_tokens = get_batch_tokens(dataset_iter, all_tokens_batches, model, encoder.cfg)
    torch.save(all_tokens, DATA_DIR / "owt_tokens_reshaped.pt")
    print(f"all_tokens.shape: {all_tokens.shape}")


# %%
# Check induction scores for 5.1 and 5.5 on synthetic data

def get_avg_prefix_score_for_head(num_examples, seq_len, prefix_len, layer, head_index):
    res = torch.zeros(num_examples)

    for i in tqdm.tqdm(range(num_examples)):
        random_tokens = torch.randperm(model.cfg.d_vocab)[:seq_len].unsqueeze(0).to(device)
        random_tokens[:, 0] = model.tokenizer.bos_token_id

        a1_pos = random.randint(1, seq_len-30)
        a2_pos = random.randint(a1_pos+prefix_len+1, seq_len-(prefix_len+1))

        random_tokens[0, a2_pos:a2_pos+prefix_len] = random_tokens[0, a1_pos:a1_pos+prefix_len]
        _, cache = model.run_with_cache(random_tokens)

        res[i] = cache['pattern', layer][0, head_index, a2_pos+prefix_len-1, a1_pos+prefix_len]

    return res.mean()

torch.manual_seed(0)
random.seed(0) 
seq_len = 128
num_examples = 400
max_prefix_len = 10
l5_h1_prefix_scores = [get_avg_prefix_score_for_head(num_examples, seq_len, prefix_len, layer=5, head_index=1) for prefix_len in range(1, max_prefix_len)]
l5_h5_prefix_scores = [get_avg_prefix_score_for_head(num_examples, seq_len, prefix_len, layer=5, head_index=5) for prefix_len in range(1, max_prefix_len)]

# %%
lines(
    [l5_h1_prefix_scores, l5_h5_prefix_scores],
    title="Avg Induction score as function of prefix length",
    xaxis="prefix length", yaxis="avg induction score", labels=["5.5", "5.1"], x=[1, 2, 3, 4, 5, 6, 7, 8, 9]
)
# %%
## Check DLA to correct inducted token for 5.1 and 5.5 on synthetic data
def get_avg_dla_for_head(num_examples, seq_len, prefix_len, layer, head_index):
    res = torch.zeros(num_examples)

    for i in tqdm.tqdm(range(num_examples)):
        random_tokens = torch.randperm(model.cfg.d_vocab)[:seq_len].unsqueeze(0).to(device)
        random_tokens[:, 0] = model.tokenizer.bos_token_id

        a1_pos = random.randint(1, seq_len-30)
        a2_pos = random.randint(a1_pos+prefix_len+1, seq_len-(prefix_len+1))

        random_tokens[0, a2_pos:a2_pos+prefix_len] = random_tokens[0, a1_pos:a1_pos+prefix_len]

        _, cache = model.run_with_cache(random_tokens)
        
        src_pos = a1_pos+prefix_len
        ans_tok = random_tokens[0, src_pos]
        logit_dir = model.W_U[:, ans_tok]
        
        dest_pos = a2_pos+prefix_len-1
        
        resid_stack, labels = cache.stack_head_results(layer=layer+1, return_labels=True, apply_ln=True, pos_slice=dest_pos)
        stack_idx = labels.index(f"L{layer}H{head_index}")
        assert labels[stack_idx] == f"L{layer}H{head_index}"
        head_result = resid_stack[stack_idx, 0, :]
        
        dla = head_result @ logit_dir

        res[i] = dla

    return res.mean()

torch.manual_seed(0)
random.seed(0) 
seq_len = 128
num_examples = 400
max_prefix_len = 10

l5_h1_dla_per_prefix = [get_avg_dla_for_head(num_examples, seq_len, prefix_len, layer=5, head_index=1) for prefix_len in range(1, max_prefix_len)]

l5_h5_dla_per_prefix = [get_avg_dla_for_head(num_examples, seq_len, prefix_len, layer=5, head_index=5) for prefix_len in range(1, max_prefix_len)]

print(l5_h1_dla_per_prefix)
print(l5_h5_dla_per_prefix)
# %%
lines(
    [l5_h5_dla_per_prefix, l5_h1_dla_per_prefix],
    title="Avg corect token DLA as function of prefix length",
    xaxis="prefix length", yaxis="avg dla score", labels=["5.5", "5.1"], x=[1, 2, 3, 4, 5, 6, 7, 8, 9]
)
# %%
### Now check on real training distribution ###
# Helper functions to check for fuzzy induction
def get_token_with_space(token: int) -> int:
    str_tok = model.to_string(token)
    new_str_tok = " " + str_tok
    try:
        new_token = model.to_single_token(new_str_tok)
    except:
        return token
    return new_token

def remove_space_from_token(token: int) -> int:
    str_tok = model.to_string(token)
    if str_tok[0] != " ": return token
    new_str_tok = str_tok[1:]
    try:
        new_token = model.to_single_token(new_str_tok)
    except:
        return token
    return new_token

def get_uppercase_token(token: int) -> int:
    str_tok = model.to_string(token)
    new_str_tok = str_tok.capitalize()
    try:
        new_token = model.to_single_token(new_str_tok)
    except:
        return token
    return new_token

def get_lowercase_token(token: int) -> int:
    str_tok = model.to_string(token)
    new_str_tok = str_tok.lower()
    try:
        new_token = model.to_single_token(new_str_tok)
    except:
        return token
    return new_token

def check_fuzzy_match(tok1: int, tok2: int) -> bool:
    return tok1 == tok2 or tok1 == get_token_with_space(tok2) or tok1 == remove_space_from_token(tok2) or tok1 == get_uppercase_token(tok2) or tok1 == get_lowercase_token(tok2)
# %%
#### Bar graph: intervene on induction examples
def check_long_prefix_by_attn(model, batch_size, num_batches, layer, head_index, attn_threshold=0.5):
    """
    For some attention head, finds examples where where the model is attending to some token above attn_threshold
    Also checks whether these examples are induction, and the corresponding prefix length
    Args:
        model: HookedTransformer
        batch_size: int
        num_batches: int
        layer: int
        head_index: int
        attn_threshold: float
    
    Returns:
        long_induction_examples: List[tuple(str, int, int, int)]: (tokens, dest_pos, src_pos, prefix_len)
        prefix_lens: List[int]
    """
    long_induction_examples = []
    prefix_lens = []
    for batch_idx in tqdm.trange(num_batches):
        tokens = all_tokens[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        _, cache = model.run_with_cache(tokens)

        patterns = cache['pattern', layer][:, head_index] # [batch, dest_pos, src_pos]
        mask = patterns > attn_threshold
        mask[:, :, 0] = False # Ignore examples where we mostly attend to BOS

        for b in range(mask.shape[0]):
            if mask[b].any():
                dest_pos, src_pos = torch.where(mask[b])
                dest_pos = dest_pos[0].item()
                src_pos = src_pos[0].item()
                
                sub_res = [tokens[b], dest_pos, src_pos]
                
                prefix_len = 0
                src_pos -= 1
                while src_pos > 0 and check_fuzzy_match(tokens[b, dest_pos], tokens[b, src_pos]):
                    dest_pos -= 1
                    src_pos -= 1
                    prefix_len += 1
                
                prefix_lens.append(prefix_len)
                sub_res.append(prefix_len)
                long_induction_examples.append(sub_res)
    
    return long_induction_examples, prefix_lens

# start with 5.1
torch.manual_seed(0)
random.seed(0)


num_batches = 50
batch_size = 32
layer = 5
head_index = 1
attn_threshold = 0.3
long_induction_examples_1, _ = check_long_prefix_by_attn(model, batch_size, num_batches, layer, head_index, attn_threshold)

# %% 
# Causally intervene on long prefix examples and show that attention drops
torch.manual_seed(0)
random.seed(0)
old_attns = []
new_attns = []
for tokens, dest_pos, src_pos, prefix_len in tqdm.tqdm(long_induction_examples_1):
    if prefix_len >= 2:
        _, cache = model.run_with_cache(tokens)
        old_attn = cache['pattern', layer][:, head_index, dest_pos, src_pos].item()
        new_tokens = tokens.clone()
        new_tokens[src_pos-2] = random.randint(0, model.cfg.d_vocab)
        _, new_cache = model.run_with_cache(new_tokens)
        new_attn = new_cache['pattern', layer][:, head_index, dest_pos, src_pos].item()
        
        old_attns.append(old_attn)
        new_attns.append(new_attn)

# %%
old_attn_means_5_1 = np.mean(old_attns)  # Example value
new_attn_means_5_1 = np.mean(new_attns)  # Example value

# %%
# compare to 5.5
head_index = 5
long_induction_examples_5, _ = check_long_prefix_by_attn(model, batch_size, num_batches, layer, head_index, attn_threshold)

# %%
torch.manual_seed(0)
random.seed(0)

old_attns = []
new_attns = []

actual_long_induction_examples = []
for tokens, dest_pos, src_pos, prefix_len in tqdm.tqdm(long_induction_examples_5):
    if prefix_len >= 2:
        _, cache = model.run_with_cache(tokens)
        old_attn = cache['pattern', layer][:, head_index, dest_pos, src_pos].item()
        new_tokens = tokens.clone()
        new_tokens[src_pos-2] = random.randint(0, model.cfg.d_vocab)
        _, new_cache = model.run_with_cache(new_tokens)
        new_attn = new_cache['pattern', layer][:, head_index, dest_pos, src_pos].item()
        
        old_attns.append(old_attn)
        new_attns.append(new_attn)
        actual_long_induction_examples.append((tokens, dest_pos, src_pos, prefix_len))  

# %%

import plotly.graph_objects as go
old_attn_means_5_5 = np.mean(old_attns)  # Example value
new_attn_means_5_5 = np.mean(new_attns)  # Example value

fig = go.Figure()

fig.add_trace(go.Bar(
    x=['Clean (long prefix)', 'Corrupted (1-prefix)'],
    y=[old_attn_means_5_1, new_attn_means_5_1],
    name='5.1',
    marker_color='red'
))

fig.add_trace(go.Bar(
    x=['Clean (long prefix)', 'Corrupted (1-prefix)'],
    y=[old_attn_means_5_5, new_attn_means_5_5],
    name='5.5',
    marker_color='blue'
))

fig.update_layout(
    barmode='group',
    title='Avg Induction Score Before and After Intervention',
    xaxis_title='Clean vs Corrupted',
    yaxis_title='Induction Score',
    legend_title='Head',
    yaxis=dict(range=[0, 1])
)

fig.show()
# %%

### Histogram: How often the heads are doing induction of varying prefix lengths
# Given attention > 0.3
attn_threshold = 0.3
num_batches = 1000
prefix_lens_list = []
long_induction_examples_list = []
for head_index in [1, 5]:
    long_induction_examples, prefix_lens = check_long_prefix_by_attn(model, batch_size, num_batches, layer, head_index, attn_threshold)
    prefix_lens_list.append(prefix_lens)
    long_induction_examples_list.append(long_induction_examples)

df = pd.DataFrame({
    "Prefix Len": prefix_lens_list[1] + prefix_lens_list[0],
    "Head": ["5.5"] * len(prefix_lens_list[1]) + ["5.1"] * len(prefix_lens_list[0])
})

fig = px.histogram(df, x="Prefix Len", color="Head",
                   barmode='overlay',
                   opacity=0.75,
                   title=f"Distribution of Prefix lengths given attn > {attn_threshold}")

fig.show()
# %%