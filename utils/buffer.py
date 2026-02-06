import torch 
import torch.nn.functional as F

import numpy as np 

from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class Chunk:
    """A standalone sequence segment for RL² training."""
    h_0: Optional[torch.Tensor]           # (num_layers, 1, h_dim)
    obs: torch.Tensor           # (seq_len, 1, obs_dim)
    prev_a: torch.Tensor        # (seq_len, 1, action_dim) for continuous and (seq_len, 1) for discrete
    prev_r: torch.Tensor        # (seq_len, 1, 1)
    prev_done: torch.Tensor     # (seq_len, 1, 1)
    act: torch.Tensor           # (seq_len, 1, action_dim) for continuous and (seq_len, 1) for discrete
    raw_act: torch.Tensor       # (seq_len, 1, action_dim) for continuous and (seq_len, 1) for discrete
    old_logp: torch.Tensor      # (seq_len, 1)
    ret: torch.Tensor           # (seq_len, 1)
    adv: torch.Tensor           # (seq_len, 1)





class RL2RolloutBuffer:
    def __init__(self, num_envs: int, device: torch.device, is_discrete: bool):
        self.num_envs = num_envs
        self.device = device
        self.is_discrete = is_discrete
        self.reset()

    
    def reset(self):
        self.obs: List[np.ndarray] = []
        self.prev_a: List[Union[int, np.ndarray]] = []
        self.prev_r: List[float] = []
        self.prev_done: List[float] = []
        self.act: List[Union[int, np.ndarray]] = [] 
        self.raw_act: List[Union[int, np.ndarray]] = []
        self.logp: List[float] = []
        self.val: List[float] = []
        self.rew: List[float] = []
        self.done: List[float] = []
        self.h: List[Optional[torch.Tensor]] = []
        self.adv_t: Optional[torch.Tensor] = None
        self.ret_t: Optional[torch.Tensor] = None

    
    def add(self, obs, prev_a, prev_r, prev_done, act, raw_act, logp, val, rew, done, h):
        self.obs.append(obs)
        self.prev_a.append(prev_a)
        self.prev_r.append(prev_r)
        self.prev_done.append(prev_done)
        self.act.append(act)
        self.raw_act.append(raw_act)
        self.logp.append(logp)
        self.val.append(val)
        self.rew.append(rew)
        self.done.append(done)
        self.h.append(h.detach() if h is not None else None)

    
    def compute_gae(self, last_values: torch.Tensor, gamma: float, lam: float):
        """
        last_values: (num_envs, ), the critic's value estimate for the next state (the bootstrap)
        """
        T = len(self.rew)
        N = self.num_envs
        vals = torch.as_tensor(np.stack(self.val), device=self.device, dtype=torch.float32).view(T, N)
        rews = torch.as_tensor(np.stack(self.rew), device=self.device, dtype=torch.float32).view(T, N)
        dones = torch.as_tensor(np.stack(self.done), device=self.device, dtype=torch.float32).view(T, N)
        
        full_vals = torch.cat([vals, last_values.view(1, N)], dim=0)
        adv = torch.zeros(T, self.num_envs, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rews[t] + gamma * mask * full_vals[t+1] - full_vals[t]
            gae = delta + gamma * lam * mask * gae #(num_envs, )
            adv[t] = gae
        
        self.adv_t = adv 
        self.ret_t = adv + vals

    

    def build_chunks(self, chunk_len: int) -> List[Chunk]:
        assert self.adv_t is not None, "Compute GAE first"
        T, N = self.adv_t.shape #(seq_len, num_envs)

        obs_t = torch.as_tensor(np.stack(self.obs), device=self.device)
        # Using .view ensures the scalar lists become (T, N, 1) for consistent slicing
        prev_r_t = torch.as_tensor(np.stack(self.prev_r), device=self.device).float().view(T, N, 1)
        prev_done_t = torch.as_tensor(np.stack(self.prev_done), device=self.device).float().view(T, N, 1)
        old_logp_t = torch.as_tensor(np.stack(self.logp), device=self.device).view(T, N, 1)

        if self.is_discrete:
            prev_a_t = torch.as_tensor(np.stack(self.prev_a), device=self.device, dtype=torch.long).view(T, N, 1)
            act_t = torch.as_tensor(np.stack(self.act), device=self.device, dtype=torch.long).view(T, N, 1)
            raw_act_t = act_t
        else:
            prev_a_t = torch.as_tensor(np.stack(self.prev_a), device=self.device)
            act_t = torch.as_tensor(np.stack(self.act), device=self.device)
            raw_act_t = torch.as_tensor(np.stack(self.raw_act), device=self.device)

        chunks = []
        for env_idx in range(N):
            for start in range(0, T, chunk_len):
                end = min(start + chunk_len, T)
                if end - start < 2: continue
                
                #h_0 = self.h[start][:, env_idx:env_idx+1, :].contiguous()
                raw_h = self.h[start]
                if raw_h is not None:
                    if raw_h.dim() == 2:
                        raw_h = raw_h.unsqueeze(0)
                    h_0 = raw_h[:, env_idx:env_idx+1, :].contiguous()
                else:
                    h_0 = None # Let combine_chunks handle the zero-init

                chunks.append(Chunk(
                    h_0=h_0,
                    obs=obs_t[start:end, env_idx].unsqueeze(1),
                    prev_a=prev_a_t[start:end, env_idx].unsqueeze(1),
                    prev_r=prev_r_t[start:end, env_idx].unsqueeze(1),
                    prev_done=prev_done_t[start:end, env_idx].unsqueeze(1),
                    act=act_t[start:end, env_idx].unsqueeze(1),
                    raw_act=raw_act_t[start:end, env_idx].unsqueeze(1),
                    old_logp=old_logp_t[start:end, env_idx].unsqueeze(1),
                    ret=self.ret_t[start:end, env_idx].unsqueeze(1),
                    adv=self.adv_t[start:end, env_idx].unsqueeze(1)
                ))
        return chunks





    

def combine_chunks(mb: List[Chunk], h_dim: int, num_layers: int = 1):
    """
    Combines a list of Chunks into batched tensors.
    Handles optional h_0 by providing zero-state defaults.
    """
    assert len(mb) > 0, "Empty minibatch provided"
    
    L = min(c.obs.shape[0] for c in mb)
    device = mb[0].obs.device
    
    def cat_attr(name):
        return torch.cat([getattr(c, name)[:L] for c in mb], dim=1)

    obs       = cat_attr('obs')
    prev_a    = cat_attr('prev_a')
    prev_r    = cat_attr('prev_r')
    prev_done = cat_attr('prev_done')
    act       = cat_attr('act')
    raw_act   = cat_attr('raw_act')
    old_logp  = cat_attr('old_logp').squeeze(-1) 
    adv       = cat_attr('adv').squeeze(-1)      
    ret       = cat_attr('ret').squeeze(-1)      
    
    # --- Robust h_0 Handling ---
    h_list = []
    for c in mb:
        if c.h_0 is not None:
            h_list.append(c.h_0)
        else:
            # Create a zero-filled state if h_0 is missing
            h_list.append(torch.zeros(num_layers, 1, h_dim, device=device))
    
    h_0 = torch.cat(h_list, dim=1)

    return obs, prev_a, prev_r, prev_done, act, raw_act, old_logp, adv, ret, h_0, L
    


    

    








