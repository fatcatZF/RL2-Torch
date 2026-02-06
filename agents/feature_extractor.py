import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 



class RL2GRUFeatureExtractor(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 h_dim: int,
                 is_discrete=True,
                 action_embed_dim=None, 
                 state_norm: nn.Module = None,
                 reward_norm: nn.Module = None, 
                 action_norm: nn.Module = None):
        super().__init__()
        self.is_discrete = is_discrete
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.h_dim = h_dim

        # Normalizers
        self.state_norm = state_norm
        self.reward_norm = reward_norm
        self.action_norm = action_norm

        #  Action Embedding Logic
        if self.is_discrete:
            if action_embed_dim is None:
                action_embed_dim = min(16, max(4, int(np.ceil(np.log2(action_dim))) * 2))
            self.action_embedder = nn.Embedding(action_dim, action_embed_dim)
            combined_input_dim = state_dim + action_embed_dim + 2
        else:
            self.action_embedder = None
            # If continuous, use action_dim directly
            combined_input_dim = state_dim + action_dim + 2

        # Processing Layers
        self.encoder = nn.Linear(combined_input_dim, h_dim)
        self.layer_norm = nn.LayerNorm(h_dim)
        self.gru = nn.GRU(h_dim, h_dim)

    
    def forward(self, state, prev_action, reward, done, h_0=None, 
                update_rms=True):
        # Normalize State
        if self.state_norm is not None:
            state = self.state_norm(state, update=update_rms)

        # Normalize reward
        if self.reward_norm is not None:
            reward = self.reward_norm(reward, update=update_rms) 

        
        if self.is_discrete and prev_action.dim() == 3:
            prev_action = prev_action.squeeze(-1)

        # Handle previous action 
        if self.is_discrete:
            # discrete actions usually aren't "normalized", but we embed them
            action_val = self.action_embedder(prev_action.long())
        else:
            # Apply normalization to continuous actions if provided
            action_val = prev_action
            if self.action_norm is not None:
                action_val = self.action_norm(action_val, update=update_rms)

       

        x = torch.cat([state, action_val, reward, done], dim=-1)
        x = self.encoder(x)
        x = self.layer_norm(x)
        gru_input = F.relu(x)

        output, next_hidden = self.gru(gru_input, h_0)
        return output, next_hidden