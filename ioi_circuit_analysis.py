# %%
from common.utils import *
# %%
pio.renderers.default = 'jupyterlab'
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

auto_encoder_runs = [
    "gpt2-small_L0_Hcat_z_lr1.20e-03_l11.80e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L1_Hcat_z_lr1.20e-03_l18.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v5",
    "gpt2-small_L2_Hcat_z_lr1.20e-03_l11.00e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v4",
    "gpt2-small_L3_Hcat_z_lr1.20e-03_l19.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L4_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v7",
    "gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L6_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L7_Hcat_z_lr1.20e-03_l11.10e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L8_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v6",
    "gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L10_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L11_Hcat_z_lr1.20e-03_l13.00e+00_ds24576_bs4096_dc3.16e-06_rsanthropic_rie25000_nr4_v9",
]
encoders = [AutoEncoder.load_from_hf(auto_encoder_run, hf_repo="ckkissane/attn-saes-gpt2-small-all-layers") for auto_encoder_run in auto_encoder_runs]
# %%
from common.torch_modules.HookedSAETransformer import HookedSAETransformer
model: HookedSAETransformer = HookedSAETransformer.from_pretrained(encoders[0].cfg["model_name"]).to(device)
# %%
from ioi_dataset import IOIDataset

N = 100
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=str(device)
)
# %%
tokens = ioi_dataset.toks
answer_tokens = torch.tensor([(correct_ans, wrong_ans) for correct_ans, wrong_ans in zip(ioi_dataset.io_tokenIDs, ioi_dataset.s_tokenIDs)]).to(device)
def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    final_logits = logits[torch.arange(logits.shape[0]), ioi_dataset.word_idx["end"], :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()

# %%
from common.torch_modules.HookedSAEConfig import HookedSAEConfig
from common.torch_modules.HookedSAE import HookedSAE
from transformer_lens.utils import download_file_from_hf

def attn_sae_cfg_to_hooked_sae_cfg(attn_sae_cfg):
    new_cfg = {
        "d_sae": attn_sae_cfg["dict_size"],
        "d_in": attn_sae_cfg["act_size"],
        "hook_name": attn_sae_cfg["act_name"],
    }
    return HookedSAEConfig.from_dict(new_cfg)

hf_repo = "ckkissane/attn-saes-gpt2-small-all-layers"

print("Attached SAEs before:", model.acts_to_saes.keys())
for auto_encoder_run in auto_encoder_runs:
    att_sae_cfg = download_file_from_hf(hf_repo, f"{auto_encoder_run}_cfg.json")
    cfg = attn_sae_cfg_to_hooked_sae_cfg(att_sae_cfg)
    
    state_dict = download_file_from_hf(hf_repo, f"{auto_encoder_run}.pt", force_is_torch=True)
    
    hooked_sae = HookedSAE(cfg)
    hooked_sae.load_state_dict(state_dict)
    model.attach_sae(hooked_sae, turn_on=False)
print("Attached SAEs after:", model.acts_to_saes.keys())
# %%
def zero_abl_attn_out(attn_out, hook, pos=None):
    if pos is not None:
        attn_out = 0.0
    else:
        attn_out[torch.arange(attn_out.shape[0]), pos, :] = 0.0
    
    return attn_out

# %%
# Now try both prepending and middle filler text
random_token_dataset = ioi_dataset.gen_flipped_prompts("ABB->CDD, BAB->DCD")
prepend_text = "It was a nice day. "
prepend_tokens = model.to_tokens(prepend_text, prepend_bos=False)
filler_text = " while the weather was nice"
filler_tokens = model.to_tokens(filler_text, prepend_bos=False)
random_tokens_prepend_and_middle_filler = []
for i in tqdm.trange(random_token_dataset.toks.shape[0]):
    new_tokens = torch.cat([prepend_tokens[0],random_token_dataset.toks[i][:random_token_dataset.word_idx["punct"][i]], filler_tokens[0], random_token_dataset.toks[i][random_token_dataset.word_idx["punct"][i]:]]) 
    random_tokens_prepend_and_middle_filler.append(new_tokens)

random_tokens_prepend_and_middle_filler = torch.stack(random_tokens_prepend_and_middle_filler).to(device)
print("Clean:", model.to_string(tokens[7]))
print("Corrupted:", model.to_string(random_tokens_prepend_and_middle_filler[7]))
# %%
and_indices = []
for i in range(tokens.shape[0]):
    if ioi_dataset.templates_by_prompt[i] == "ABBA":
        and_indices.append(ioi_dataset.word_idx["S1-1"][i])
    else:
        and_indices.append(ioi_dataset.word_idx["S1+1"][i])

and_indices = torch.tensor(and_indices)

# %%
def replace_and_token(tokens, and_indices, new_token):
    new_tokens = tokens.clone()
    new_tokens[torch.arange(new_tokens.shape[0]), and_indices] = new_token
    return new_tokens

# %% 
# Turn on error terms
for layer in range(model.cfg.n_layers):
    act_name = utils.get_act_name('z', layer)
    model.acts_to_saes[act_name].cfg.use_error_term = True
    if layer == 5:
        print(model.acts_to_saes[act_name].cfg)
        
model.turn_saes_on()
model.get_saes_status()
# %%
def ablate_sae_feature(sae_acts, hook, pos, feature_id):
    if pos is None:
        sae_acts[:, :, feature_id] = 0.
    else:
        sae_acts[:, pos, feature_id] = 0.
    return sae_acts

def ablate_sae_error(sae_error, hook, pos):
    if pos is None:
        sae_error = 0.
    else:
        sae_error[:, pos, ...] = 0.

layer = 5
_, cache = model.run_with_cache_with_saes(tokens, act_names=utils.get_act_name('z', layer))
s2_pos = ioi_dataset.word_idx["S2"]
sae_acts = cache[utils.get_act_name('z', layer) + ".hook_sae_acts_post"][torch.arange(tokens.shape[0]), s2_pos, :]

live_feature_mask = sae_acts > 0

live_feature_union = live_feature_mask.any(dim=0)
print("Total live features (all prompts)", live_feature_union.sum())

hooked_encoder = model.acts_to_saes[utils.get_act_name('z', layer)]
all_live_features = torch.arange(hooked_encoder.cfg.d_sae)[live_feature_union.cpu()]

causal_effects = torch.zeros((tokens.shape[0], all_live_features.shape[0]+1))
fid_to_idx = {fid.item(): idx for idx, fid in enumerate(all_live_features)}


clean_logits = model(tokens)
clean_per_prompt_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens, per_prompt=True)

abl_layer, abl_pos  = layer, s2_pos
for feature_id in tqdm.tqdm(all_live_features):
    feature_id = feature_id.item()
    abl_feature_logits = model.run_with_hooks_with_saes(
        tokens,
        return_type="logits",
        act_names=utils.get_act_name('z', abl_layer),
        fwd_hooks=[(utils.get_act_name('z', abl_layer) + ".hook_sae_acts_post", partial(ablate_sae_feature, pos=abl_pos, feature_id=feature_id))]
    ) # [batch, seq, vocab]
    
    abl_feature_logit_diff = logits_to_ave_logit_diff(abl_feature_logits, answer_tokens, per_prompt=True) # [batch]
    causal_effects[:, fid_to_idx[feature_id]] = abl_feature_logit_diff - clean_per_prompt_logit_diff

# Compare to error term
abl_error_term_logits = model.run_with_hooks_with_saes(
    tokens,
    return_type="logits",
    act_names=utils.get_act_name('z', abl_layer),
    fwd_hooks=[(utils.get_act_name('z', abl_layer) + ".hook_sae_error", partial(ablate_sae_error, pos=abl_pos))]
)
abl_error_term_logits_diff = logits_to_ave_logit_diff(abl_error_term_logits, answer_tokens, per_prompt=True)
causal_effects[:, -1] = abl_error_term_logits_diff - clean_per_prompt_logit_diff
# %%
abba_indices = [i for i, template in enumerate(ioi_dataset.templates_by_prompt) if template == "ABBA"]
baba_indices = [i for i, template in enumerate(ioi_dataset.templates_by_prompt) if template == "BABA"]

# %%
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

data = pd.DataFrame({
    "Feature": list(map(str, all_live_features.tolist())) + ["error_term"],
    "ABBA": causal_effects[abba_indices].mean(0),
    "BABA": causal_effects[baba_indices].mean(0)
})

data_melted = pd.melt(data, id_vars="Feature", var_name="Template", value_name="Change in Logit Diff")
data_melted_sorted = data_melted.sort_values("Change in Logit Diff").head(15)
data_melted_sorted["Change in Logit Diff"] = data_melted_sorted["Change in Logit Diff"].apply(lambda x: round(x, 3))

fig = make_subplots(
    rows=2, cols=2,
    column_widths=[0.4, 0.6],
    row_heights=[0.2, 0.8],
    specs=[[{"type": "table", "rowspan": 2}, {"type": "xy"}],
           [None, {"type": "xy"}]],
    vertical_spacing=0.02
)

fig.add_trace(
    go.Table(
        header=dict(values=["Feature Id", "Template", "Δ logit diff"]),
        cells=dict(values=[data_melted_sorted["Feature"], data_melted_sorted["Template"], data_melted_sorted["Change in Logit Diff"]])
    ),
    row=1, col=1
)

histogram_fig = px.histogram(
    data_melted,
    x="Change in Logit Diff",
    color="Template",
    barmode="group"
)

for trace in histogram_fig.data:
    fig.add_trace(trace, row=2, col=2)

box_fig = px.box(data_melted, x="Change in Logit Diff", color="Template")

for trace in box_fig.data:
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=2)

fig.update_xaxes(title_text="Δ logit diff", row=2, col=2)
fig.update_yaxes(title_text="Count", row=2, col=2)
fig.update_xaxes(title_text="", row=1, col=2, showticklabels=False)
fig.update_yaxes(title_text="", row=1, col=2)

fig.update_layout(
    title_text=f"Avg Change in logit diff when ablating L{layer} SAE features at S2 pos (including error term)",
)

fig.show()
# %%
# Nosing experiments (Figure 5)
def patch_attn_out(attn_out, hook, corrupted_cache, pos=None, corr_offset=0):
    if pos is None:
        attn_out = corrupted_cache[hook.name]
    else:
        attn_out[torch.arange(attn_out.shape[0]), pos, :] = corrupted_cache[hook.name][torch.arange(attn_out.shape[0]), pos+corr_offset, :]
    
    return attn_out


s2_pos = ioi_dataset.word_idx["S2"]

layers = [5, 6]
records = []
and_replacement = ' alongside'

new_token = model.to_single_token(and_replacement)
clean_tokens = ioi_dataset.toks.clone()
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_per_prompt_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens, per_prompt=True)

corrupted_tokens = replace_and_token(tokens, and_indices, new_token)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
corrupted_per_prompt_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens, per_prompt=True)

noised_logits = model.run_with_hooks(
    clean_tokens,
    fwd_hooks=[(utils.get_act_name('attn_out', layer), partial(patch_attn_out, corrupted_cache=corrupted_cache, pos=s2_pos)) for layer in layers]
)
noised_along_side_logit_diff_per_prompt = logits_to_ave_logit_diff(noised_logits, answer_tokens, per_prompt=True)

# %%
print("Clean:", model.to_string(tokens[2]))
print("Corrupted:", model.to_string(corrupted_tokens[2]))
# %%
corrupted_tokens = random_tokens_prepend_and_middle_filler.clone()
corr_offset=prepend_tokens.shape[1]+filler_tokens.shape[1]

records = []

clean_tokens = tokens.clone()
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_logit_diff_per_prompt = logits_to_ave_logit_diff(clean_logits, answer_tokens, per_prompt=True)

corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
corrupted_logit_diff_per_prompt = logits_to_ave_logit_diff(corrupted_logits, answer_tokens, per_prompt=True)


noised_logits = model.run_with_hooks(
    clean_tokens,
    return_type="logits",
    fwd_hooks=[
        (utils.get_act_name('attn_out', layer), partial(patch_attn_out, corrupted_cache=corrupted_cache, pos=s2_pos, corr_offset=corr_offset)) for layer in layers
    ]
)

noised_filler_logit_diff_per_prompt = logits_to_ave_logit_diff(noised_logits, answer_tokens, per_prompt=True)

zero_abl_logits = model.run_with_hooks(
    clean_tokens,
    return_type="logits",
    fwd_hooks=[
        (utils.get_act_name('attn_out', layer), partial(zero_abl_attn_out, pos=s2_pos)) for layer in layers
    ]
)

zero_abl_logit_diff_per_prompt = logits_to_ave_logit_diff(zero_abl_logits, answer_tokens, per_prompt=True)

interventions = ["Clean", "Zero Abl", "Noised: rand toks + filler", "Noised: alongside"]
logit_diffs = [clean_per_prompt_logit_diff, zero_abl_logit_diff_per_prompt, noised_filler_logit_diff_per_prompt, noised_along_side_logit_diff_per_prompt]

mean_diffs = [logit_diff.mean().item() for logit_diff in logit_diffs]
std_diffs = [logit_diff.std().item() for logit_diff in logit_diffs]

data = {
    'Intervention': interventions,
    'Mean Logit Diff': mean_diffs,
    'Std Logit Diff': std_diffs
}

df = pd.DataFrame(data)

clean_mean_logit_diff = df.loc[df['Intervention'] == 'Clean', 'Mean Logit Diff'].values[0]
zero_abl_mean_logit_diff = df.loc[df['Intervention'] == 'Zero Abl', 'Mean Logit Diff'].values[0]

df_filtered = df.tail(2)

x_axis = df_filtered['Intervention']
y_data = df_filtered['Mean Logit Diff']
error_y_data = df_filtered['Std Logit Diff']
new_labels = ["rand toks + filler", "alongside"]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=[new_labels[0]],
    y=[y_data.iloc[0]],
    error_y=dict(
        type='data',
        array=[error_y_data.iloc[0]],
        visible=True
    ),
    marker_color='blue',
    showlegend=False 
))

fig.add_trace(go.Bar(
    x=[new_labels[1]],
    y=[y_data.iloc[1]],
    error_y=dict(
        type='data',
        array=[error_y_data.iloc[1]],
        visible=True
    ),
    marker_color='red',
    showlegend=False
))

fig.add_shape(
    type='line',
    x0=-0.5,
    x1=1.5,
    y0=clean_mean_logit_diff,
    y1=clean_mean_logit_diff,
    line=dict(
        color='black',
        width=1,
        dash='dot'
    )
)

fig.add_annotation(
    xref="paper",
    x=1.02,
    y=clean_mean_logit_diff,
    xanchor="left",
    text="Clean",
    showarrow=False,
    font=dict(
        color="black",
        size=12
    )
)

fig.add_shape(
    type='line',
    x0=-0.5,
    x1=1.5,
    y0=zero_abl_mean_logit_diff,
    y1=zero_abl_mean_logit_diff,
    line=dict(
        color='black',
        width=1,
        dash='dot'
    )
)

fig.add_annotation(
    xref="paper",
    x=1.02,
    y=zero_abl_mean_logit_diff,
    xanchor="left",
    text="Zero Abl",
    showarrow=False,
    font=dict(
        color="black",
        size=12
    )
)

fig.update_layout(
    title_text="IOI Logit Difference",
    xaxis_title_text="Intervention",
    yaxis_title_text="Logit Diff",
    template="plotly_white",
    xaxis_showgrid=True, 
    yaxis_showgrid=True, 
    xaxis_showline=True, 
    yaxis_showline=True, 
    xaxis_linecolor='black', 
    yaxis_linecolor='black',
)

fig.show()
# %%