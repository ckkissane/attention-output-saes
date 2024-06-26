# %%
from common.visualize_utils import *

# %%
auto_encoder_run = "gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9"
encoder = AutoEncoder.load_from_hf(auto_encoder_run, hf_repo="ckkissane/attn-saes-gpt2-small-all-layers")
model = HookedTransformer.from_pretrained(encoder.cfg["model_name"]).to(DTYPES[encoder.cfg["enc_dtype"]]).to(encoder.cfg["device"])

# %%
@dataclass
class DecoderWeightsDistribution:
  n_heads: int
  allocation_by_head: List[float]

def get_all_decoder_weights_distributions(encoder: AutoEncoder, model: nn.Module) -> Float[Tensor, "n_features n_head"]:

    att_blocks = einops.rearrange(encoder.W_dec, "n_features (n_head d_head) -> n_features n_head d_head", n_head=model.cfg.n_heads)
    decoder_weights_distribution = att_blocks.norm(dim=-1) / att_blocks.norm(dim=-1).sum(dim=-1, keepdim=True)

    return decoder_weights_distribution
# %%
import tqdm

topk = 10
all_tokens = load_all_tokens()
max_batch_size = 128
total_batch_size = int(1e6) // encoder.cfg["seq_len"]

auto_encoder_runs = {
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
    11: "gpt2-small_L11_Hcat_z_lr1.20e-03_l13.00e+00_ds24576_bs4096_dc3.16e-06_rsanthropic_rie25000_nr4_v9"
}

for i in tqdm.tqdm(range(model.cfg.n_layers)):
    encoder = AutoEncoder.load_from_hf(auto_encoder_runs[i], hf_repo="ckkissane/attn-saes-gpt2-small-all-layers")

    distrib = get_all_decoder_weights_distributions(encoder, model)
    top_features_per_head = {
        head: torch.topk(distrib[:, head], topk).indices
        for head in range(model.cfg.n_heads)
    }

    for j in tqdm.tqdm(range(model.cfg.n_heads)):
        html = ""
        html += f"<h1>H{i}.{j} top features</h1>"
        features = top_features_per_head[j][:topk].tolist()

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
        max_act_indices_list = get_max_act_indices(
            encoder,
            model,
            tokens=all_tokens[:total_batch_size], # Int[Tensor, "batch seq"]
            feature_idx = features,
            max_batch_size=max_batch_size,
        )

        for k in range(topk):
            html += f"<h2>Top feature {k} in H{i}.{j}: (feature {features[k]}</h2>"
            html += get_feature_card_from_feature_data(all_feature_data, features, k, layer_ix=i,
                                                       max_batch_size=max_batch_size, total_batch_size=total_batch_size,
                                                       max_act_indices=max_act_indices_list,
                                                       model=model, encoder=encoder, all_tokens=all_tokens, display=False,)

        with open(f"cards/top_features_{i}_{j}.html", "w") as f:
            f.write(html)
# %%
