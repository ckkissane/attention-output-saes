# %%
from common.utils import *
import random
from transformer_lens import HookedTransformer
import pprint
from datasets import load_dataset

# %%
# Setup config / model
default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 384,
    "lr": 0.0012,
    "num_tokens": int(1e8),
    "l1_coeff": 0.5,
    "beta1": 0.0,
    "beta2": 0.9999,
    "dict_mult": 32,
    "seq_len": 128,
    "enc_dtype":"fp32",
    "model_name": "gpt2-small",
    "site": "z",
    "layer": 7,
    "device": "cuda",
    "reinit": "reinit",
    "head": "cat",
    "concat_heads": True,
    "resample_scheme": "anthropic",
    "anthropic_neuron_resample_scale": 0.2,
    "dead_direction_cutoff": 1e-6,
    "re_init_every": 100_000, 
    "anthropic_resample_last": 100_000,
    "resample_factor": 0.01, 
    "num_resamples": 4,
    "wandb_project_name": "your_project_name",
    "wandb_entity": "your_entity_name",
    "save_state_dict_every": 50000,
    "b_dec_init": "zeros", 
    "sched_type": "cosine_warmup",
    "sched_epochs": 50*20, 
    "sched_lr_factor": 0.1, 
    "sched_warmup_epochs": 50*20,
    "sched_finish": True,
    "anthropic_resample_batches": 100, 
    "eval_every": 1000,
}
cfg = arg_parse_update_cfg(default_cfg)

SEED = cfg["seed"]
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(True)

model = HookedTransformer.from_pretrained(cfg["model_name"]).to(DTYPES[cfg["enc_dtype"]]).to(cfg["device"])

site_to_size = {
    "mlp_out": model.cfg.d_model,
    "post": model.cfg.d_mlp,
    "resid_pre": model.cfg.d_model,
    "resid_mid": model.cfg.d_model,
    "resid_post": model.cfg.d_model,
    "z": model.cfg.d_head,
}

def post_init_cfg(cfg):
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["buffer_batches"] = cfg["buffer_size"] // (cfg["seq_len"])
    cfg["act_name"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["act_size"] = site_to_size[cfg["site"]]
    if cfg["concat_heads"]:
        assert cfg["site"] in ["k", "q", "v", "z"], "concat_heads=True. Must choose a hook with a head dim"
        cfg["act_size"] = site_to_size[cfg["site"]] * model.cfg.n_heads
    cfg["dict_size"] = cfg["act_size"] * cfg["dict_mult"]
    cfg["name"] = f"{cfg['model_name']}_{cfg['layer']}_{cfg['dict_size']}_{cfg['site']}"
    assert cfg["resample_scheme"] in ["anthropic", "reinit"], "resample_scheme must be one of 'anthropic', 'reinit'"
    assert cfg["sched_type"].lower() in ["none", "cosine_warmup"], f"Unknown sched_type {cfg['sched_type']}"
    assert cfg["anthropic_resample_batches"] * cfg["batch_size"] >= cfg["dict_size"], "We are sampling with replacement, so anthropic_resample_batches * batch_size must be >= dict_size"
post_init_cfg(cfg)
pprint.pprint(cfg)
# %%
dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
dataset_iter = iter(dataset)
all_tokens = None
# %%
encoder = AutoEncoder(cfg)
buffer = StreamingBuffer(model, dataset_iter, cfg)

if cfg["b_dec_init"] == "geometric_mean_init":
    encoder.initialize_b_dec_with_geometric_median(buffer)
# %%
try:
    wandb.init(project=cfg["wandb_project_name"], name=f"{encoder.get_checkpoint_name()}", entity=cfg["wandb_entity"])
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    
    sched = None if cfg["sched_type"].lower()=="none" else torch.optim.lr_scheduler.CosineAnnealingLR(encoder_optim, T_max=cfg["sched_epochs"], eta_min=cfg["lr"]*cfg["sched_lr_factor"])
    if sched is not None: 
        for _ in range(cfg["sched_warmup_epochs"]):
            sched.step()
            
    recons_scores = []
    act_freq_scores_list = []
    running_frequency_counter = torch.zeros(size=(cfg["dict_size"],)).long().cpu()
    total = 0
    resample_count = 0
    tokens_seen = 0
    for i in tqdm.trange(num_batches, disable=True):
        acts = buffer.next()
        loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
        loss.backward()
        encoder.make_decoder_weights_and_grad_unit_norm()
        encoder_optim.step()
        if sched is not None and (not cfg["sched_finish"] or i < cfg["sched_epochs"] or sched.get_last_lr()[0]+1e-9 < cfg["lr"]):
            sched.step()
        encoder_optim.zero_grad()
        loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
        # update frequency counter
        did_fire = torch.count_nonzero(mid_acts, dim=0).cpu()
        running_frequency_counter += did_fire
        total += mid_acts.shape[0]
        tokens_seen += mid_acts.shape[0]
        
        
        if (i) % 100 == 0:
            wandb.log(loss_dict)
        if (i) % cfg["eval_every"] == 0:
            x = (get_recons_loss(model, all_tokens, num_batches=5, cfg=cfg, local_encoder=encoder, dataset_iter=dataset_iter))
            recons_scores.append(x[0])
            freqs, num_eval_tokens = get_freqs(model, all_tokens, num_batches=5, cfg=cfg, local_encoder=encoder, disable_tqdm=True, dataset_iter=dataset_iter)
            act_freq_scores_list.append(freqs)
            avg_l0_norm = torch.count_nonzero(mid_acts, dim=-1).float().mean()
            wandb.log({
                "recons_score": x[0],
                "dead": (freqs==0).float().mean().item(),
                "below_1e-6": (freqs<1e-6).float().mean().item(),
                "below_1e-5": (freqs<1e-5).float().mean().item(),
                "L0_norm": avg_l0_norm.item(),
                "num_eval_tokens": num_eval_tokens,
                "tokens_seen": tokens_seen,
                "lr": encoder_optim.param_groups[0]["lr"],
                "ce_diff": x[4]
            })

        if (i+1) % cfg["save_state_dict_every"] == 0:
            try:
                encoder.save()
            except Exception as e:
                print("Failed to save state dict!" + str(e))
        if ((i+1) % cfg["re_init_every"]) == 0 and cfg["reinit"]:
            freqs = running_frequency_counter / total
            to_be_reset_bool = freqs<=cfg["dead_direction_cutoff"]
            to_be_reset = torch.arange(encoder.d_hidden, device=encoder.device)[to_be_reset_bool]
            if to_be_reset.numel() > 0 and resample_count < cfg["num_resamples"]:
                if cfg['resample_scheme'] == "anthropic":
                    anthropic_resample(buffer, to_be_reset_bool, encoder)
                elif cfg['resample_scheme'] == "reinit":
                    re_init(to_be_reset, encoder)
                reset_optimizer(to_be_reset, encoder_optim, cfg)
                if sched is not None:
                    max_iters = 10**7
                    while sched.get_last_lr()[0] > encoder.cfg["lr"] * encoder.cfg["sched_lr_factor"] + 1e-9:
                        sched.step()
                        max_iters -= 1
                        if max_iters == 0:
                            raise ValueError("Too many iterations -- sched is messed up")
            fig = px.histogram(utils.to_numpy(freqs.log10()), title="Log Frequency of Features", histnorm='percent')
            wandb.log({
                "frequency_histogram": fig,
                "total_freq_tokens": total,
                "num_neurons_resampled": to_be_reset.shape[0],
            })
            # reset the counter
            running_frequency_counter.zero_()
            total = 0
            resample_count += 1
        if (i + cfg["anthropic_resample_last"]) % cfg["re_init_every"] == 0:
            running_frequency_counter.zero_()
            total = 0
finally:
    encoder.save()
# %%
