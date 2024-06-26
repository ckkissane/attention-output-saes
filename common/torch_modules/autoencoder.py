import json
from geom_median.torch import compute_geometric_median
from pathlib import Path
import pprint
import os
from transformer_lens.utils import download_file_from_hf
from torch import nn
import torch
import torch.nn.functional as F

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")

class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.dtype = dtype
        self.device = cfg["device"]
        

        self.version = 0
        self.to(cfg["device"])
    
    def forward(self, x, per_token=False):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc) # [batch_size, d_hidden]
        x_reconstruct = acts @ self.W_dec + self.b_dec # [batch_size, act_size]
        if per_token:
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1) # [batch_size]
            l1_loss = self.l1_coeff * (acts.float().abs().sum(dim=-1)) # [batch_size]
            loss = l2_loss + l1_loss # [batch_size]
        else:
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0) # []
            l1_loss = self.l1_coeff * (acts.float().abs().sum(dim=-1).mean(dim=0)) # []
            loss = l2_loss + l1_loss # []
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed
    
    def get_version(self):
        res = self.version
        self.version += 1
        return res
    
    def get_checkpoint_name(self):
        return f"{self.cfg['model_name']}_L{self.cfg['layer']}_H{self.cfg['head']}_{self.cfg['site']}_lr{self.cfg['lr']:.2e}_l1{self.cfg['l1_coeff']:.2e}_ds{self.cfg['dict_size']}_bs{self.cfg['batch_size']}_dc{self.cfg['dead_direction_cutoff']:.2e}_rs{self.cfg['resample_scheme']}_rie{self.cfg['re_init_every']}_nr{self.cfg['num_resamples']}"

    def save(self, save_dir: Path = SAVE_DIR):
        version = self.get_version()
        filename = self.get_checkpoint_name() + f"_v{version}"
        torch.save(self.state_dict(), SAVE_DIR / (filename+".pt"))

        try:
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
        except:
            pass
        with open(SAVE_DIR / (filename+"_cfg.json"), "w") as f:
            json.dump(self.cfg, f)
        print("Saved as version", version)
    
    @classmethod
    def load(cls, version):
        cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
        return self

    @classmethod
    def load_from_hf(cls, version, hf_repo="ckkissane/tinystories-1M-SAES", verbose=False):
        """
        Loads the saved autoencoder from HuggingFace. 
        """
        
        cfg = download_file_from_hf(hf_repo, f"{version}_cfg.json")
        if verbose:
            pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(download_file_from_hf(hf_repo, f"{version}.pt", force_is_torch=True))
        return self
    
    @torch.no_grad()
    def initialize_b_dec_with_geometric_median(self, activation_store, num_batches=50, maxiter=100_000):
        
        previous_b_dec = self.b_dec.clone()
        
        activations_list = []
        for _ in range(num_batches): #
            activations = activation_store.next()
            activations_list.append(activations)
            
        all_activations = torch.concat(activations_list, dim=0)
        out = compute_geometric_median(
                all_activations.detach().cpu(), 
                skip_typechecks=True, 
                maxiter=maxiter, per_component=True).median
        out = torch.tensor(out, dtype=self.dtype, device=self.device)
        
        def get_sum_of_distances(b_dec):
            b_dec_expanded = b_dec.unsqueeze(0)
            squared_diff = torch.pow(all_activations - b_dec_expanded, 2)
            distances = torch.sqrt(torch.sum(squared_diff, dim=1))
            total_sum_of_distances = torch.sum(distances)
            return total_sum_of_distances

        previous_distances = get_sum_of_distances(previous_b_dec)
        distances = get_sum_of_distances(out)
        
        print("Reinitializing b_dec geometric median of activations")
        print(f"Previous distances: {previous_distances.median().item()}")
        print(f"New distances: {distances.mean().item()}")
        
        self.b_dec.data = out