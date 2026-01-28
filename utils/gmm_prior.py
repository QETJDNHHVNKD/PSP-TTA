
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def _logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim)


@dataclass
class DiagGMMParams:
    # All tensors are torch tensors on the correct device
    weights: torch.Tensor        
    means: torch.Tensor         
    vars: torch.Tensor           iagonal variances)


@dataclass
class OrgStats:
    mean: torch.Tensor          
    std: torch.Tensor           


class GMMPrior(torch.nn.Module):

    def __init__(
        self,
        org_gmms: Dict[int, DiagGMMParams],
        org_stats: Dict[int, OrgStats],
        org_mix_weights: Optional[Dict[int, float]] = None,
        eps_std: float = 1e-6,
    ):
        super().__init__()
        self.org_ids = sorted(list(org_gmms.keys()))
        self.eps_std = eps_std


        self._org_stats_mean = {}
        self._org_stats_std = {}
        self._org_gmm = {}

        for oid in self.org_ids:
            st = org_stats[oid]
            gm = org_gmms[oid]

            self.register_buffer(f"mean_{oid}", st.mean)
            self.register_buffer(f"std_{oid}", st.std)

            self.register_buffer(f"w_{oid}", gm.weights)
            self.register_buffer(f"mu_{oid}", gm.means)
            self.register_buffer(f"var_{oid}", gm.vars)

            self._org_stats_mean[oid] = f"mean_{oid}"
            self._org_stats_std[oid] = f"std_{oid}"
            self._org_gmm[oid] = (f"w_{oid}", f"mu_{oid}", f"var_{oid}")

        if org_mix_weights is None:
            org_mix_weights = {oid: 1.0 / len(self.org_ids) for oid in self.org_ids}

        mix = torch.tensor([org_mix_weights[oid] for oid in self.org_ids], dtype=torch.float32)
        mix = mix / mix.sum()
        self.register_buffer("org_mix", mix)  

    @staticmethod
    def _diag_gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:

        D = x.shape[-1]
        x_ = x.unsqueeze(1)  
        log_det = torch.log(var).sum(dim=-1)  
        quad = ((x_ - mean) ** 2 / var).sum(dim=-1)  
        return -0.5 * (quad + log_det + D * math.log(2 * math.pi))

    def _standardize(self, z: torch.Tensor, organ_id: int) -> torch.Tensor:
        mean = getattr(self, self._org_stats_mean[organ_id])
        std = getattr(self, self._org_stats_std[organ_id]).clamp_min(self.eps_std)
        return (z - mean) / std

    def log_prob_conditional(self, z: torch.Tensor, organ_id: int) -> torch.Tensor:
        zt = self._standardize(z, organ_id)
        w_name, mu_name, var_name = self._org_gmm[organ_id]
        w = getattr(self, w_name)    
        mu = getattr(self, mu_name)   
        var = getattr(self, var_name) 
        # var = var.clamp_min(1e-4)  
        var = var.clamp_min(1e-3) 


        # log_comp = self._diag_gaussian_logprob(zt, mu, var) 
        # log_mix = torch.log(w.clamp_min(1e-12)).unsqueeze(0)
        # # return _logsumexp(log_mix + log_comp, dim=1)         
        log_comp = self._diag_gaussian_logprob(zt, mu, var)  
        log_mix = torch.log(w.clamp_min(1e-12)).unsqueeze(0) 

       
        std = getattr(self, self._org_stats_std[organ_id]).clamp_min(self.eps_std) 
        log_jac = -torch.log(std).sum()  

        return _logsumexp(log_mix + log_comp, dim=1) + log_jac 

    def log_prob_mix(self, z: torch.Tensor) -> torch.Tensor:

        logps = []
        for oid in self.org_ids:
            logps.append(self.log_prob_conditional(z, oid)) 
        logps = torch.stack(logps, dim=1)  
        log_pi = torch.log(self.org_mix.clamp_min(1e-12)).unsqueeze(0) 
        return _logsumexp(log_pi + logps, dim=1)  

    def nll(
        self,
        z: torch.Tensor,
        organ_id: Optional[Union[int, torch.Tensor]] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
   
        if organ_id is None:
            lp = self.log_prob_mix(z)
        elif isinstance(organ_id, int):
            lp = self.log_prob_conditional(z, organ_id)
        else:
    
            organ_id = organ_id.to(z.device).long()
            out = torch.empty((z.shape[0],), device=z.device, dtype=z.dtype)
            for oid in organ_id.unique().tolist():
                idx = (organ_id == oid)
                out[idx] = self.log_prob_conditional(z[idx], int(oid))
            lp = out

        nll = -lp
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        return nll  # "none"

    def save(self, path: str):
        payload = {
            "org_ids": self.org_ids,
            "org_mix": self.org_mix.detach().cpu(),
            "params": {},
        }
        for oid in self.org_ids:
            payload["params"][oid] = {
                "mean": getattr(self, self._org_stats_mean[oid]).detach().cpu(),
                "std": getattr(self, self._org_stats_std[oid]).detach().cpu(),
                "weights": getattr(self, self._org_gmm[oid][0]).detach().cpu(),
                "means": getattr(self, self._org_gmm[oid][1]).detach().cpu(),
                "vars": getattr(self, self._org_gmm[oid][2]).detach().cpu(),
            }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "GMMPrior":
        ckpt = torch.load(path, map_location=device)
        org_ids = ckpt["org_ids"]
        org_mix = ckpt["org_mix"]

        org_gmms = {}
        org_stats = {}
        for oid in org_ids:
            p = ckpt["params"][oid]
            org_stats[oid] = OrgStats(mean=p["mean"].to(device), std=p["std"].to(device))
            org_gmms[oid] = DiagGMMParams(
                weights=p["weights"].to(device),
                means=p["means"].to(device),
                vars=p["vars"].to(device),
            )

        obj = cls(org_gmms=org_gmms, org_stats=org_stats, org_mix_weights=None)
        obj.org_mix.data = org_mix.to(device)
        return obj


def extract_z_shape_from_zraw(z_raw: torch.Tensor, renderer) -> torch.Tensor:
    p, cx, cy, r0, a, b = renderer.decode(z_raw)
    return torch.cat([r0, a, b], dim=1)
