# %% 
import torch
from common.visualize_utils import load_all_tokens, default_encoder, default_model, get_feature_card_from_feature_data, get_max_act_indices, get_seq_data

torch.set_grad_enabled(False)

all_tokens = load_all_tokens()

layer_ix = 9
encoder = default_encoder(layer_ix)
model = default_model(encoder)

# %%
max_batch_size = 128
total_batch_size = int(1e6) // encoder.cfg["seq_len"]
features = list(range(30))

# %%
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

max_act_indices = get_max_act_indices(
    encoder,
    model,
    tokens=all_tokens[:total_batch_size],
    feature_idx = features,
    max_batch_size=max_batch_size,
) 

# %%
sub_idx = 19
print(f"L {layer_ix} Feature {features[sub_idx]}")
get_feature_card_from_feature_data(all_feature_data, features, sub_idx, layer_ix,
                                       max_batch_size=max_batch_size, total_batch_size=total_batch_size,
                                       max_act_indices=max_act_indices,
                                       model=model, encoder=encoder, all_tokens=all_tokens, display=True)

# %%