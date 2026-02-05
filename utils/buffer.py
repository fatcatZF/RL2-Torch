import torch 

from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    # RNN state at the beginning of this chunk
    h_0: Optional[torch.Tensor] 
    
    # Inputs to the feature extractor
    obs: torch.Tensor
    prev_a: torch.Tensor
    prev_r: torch.Tensor
    prev_done: torch.Tensor
    
    # Actions
    act: torch.Tensor      # The squashed action (sent to Env)
    raw_u: torch.Tensor    # The raw Gaussian sample (used for PPO update)
    
    # Training targets/metrics
    old_logp: torch.Tensor
    ret: torch.Tensor
    adv: torch.Tensor