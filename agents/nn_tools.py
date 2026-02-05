import torch 
import torch.nn as nn 
import numpy as np 


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization. 
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, out_dim, hidden_dims=(64, 64), activation=nn.Tanh, last_std=0.01):
    """
    Creates an MLP with orthogonal initialization.
    - activation: Tanh is usually preferred over ReLU for PPO policies.
    - last_std: Small value (0.01) for the last layer helps keep initial 
                actions/gradients small and stable.
    """
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(layer_init(nn.Linear(prev, h)))
        layers.append(activation())
        prev = h
    
    # Last layer with custom std
    layers.append(layer_init(nn.Linear(prev, out_dim), std=last_std))
    return nn.Sequential(*layers)