import torch 
import torch.nn as nn 
from .nn_tools import mlp 

class RL2ActorCritic(nn.Module):
    def __init__(self,
                 feat_extractor: nn.Module,
                 actor_mlp = (64,),
                 critic_mlp = (64,),
                 ):
        super().__init__()
        # Inject the feature extractor
        self.feat_extractor = feat_extractor

        # Infer properties directly from the extractor
        # This ensures there is never a mismatch between the feature extractor and the Heads
        self.is_discrete = feat_extractor.is_discrete
        self.action_dim = feat_extractor.action_dim
        self.h_dim = feat_extractor.h_dim

        # Build heads using inferred dimensions
        if self.is_discrete:
            self.actor_head = mlp(self.h_dim, self.action_dim, hidden_dims=actor_mlp)
        else:
            self.actor_head = mlp(self.h_dim, 2*self.action_dim, hidden_dims=actor_mlp)

        self.critic_head = mlp(self.h_dim, 1, hidden_dims=critic_mlp)


    
    def forward(self, state, prev_action, reward, done, h_0=None, update_rms=True):
        output, h_n = self.feat_extractor(
            state, prev_action, reward, done, h_0, update_rms=update_rms
        )

        values = self.critic_head(output)
        raw_actor_output = self.actor_head(output)

        if self.is_discrete:
            return raw_actor_output, values, h_n
        
        else:
            mu, log_std = torch.chunk(raw_actor_output, 2, dim=-1)
            # Stability clamps for continuous PPO
            # mu = torch.clamp(mu, -5.0, 5.0)
            log_std = torch.clamp(log_std, -20, 2)
            return (mu, log_std), values, h_n



