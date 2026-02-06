import torch 
import torch.nn as nn 
import numpy as np 
from typing import Callable, List, Dict, Any



@torch.no_grad()
def run_meta_eval(
    model: torch.nn.Module,
    envs: Any, 
    device: torch.device,
    is_discrete: bool,
    max_steps: int, # The maximum safety cutoff
    action_dim: int
) -> float:
    """
    Evaluates the RL² agent. Ends early if all environments finish 
    one full episode (or meta-episode).
    """
    model.eval()
    num_envs = envs.num_envs
    obs, _ = envs.reset()
    
    # RL² Context
    h = None
    if is_discrete:
        prev_a = np.zeros(num_envs, dtype=np.int64)
    else:
        prev_a = np.zeros((num_envs, action_dim), dtype=np.float32)
    prev_r = np.zeros(num_envs, dtype=np.float32)
    prev_done = np.ones(num_envs, dtype=np.float32)
    
    # Evaluation Tracking
    total_rewards = np.zeros(num_envs)
    env_finished = np.zeros(num_envs, dtype=bool) # Track which envs hit 'done'

    for t in range(max_steps):
        # Convert to tensors
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        pa_t = torch.as_tensor(prev_a, device=device)
        pr_t = torch.as_tensor(prev_r, device=device).view(num_envs, 1)
        pd_t = torch.as_tensor(prev_done, device=device).view(num_envs, 1)

        # Forward pass (Using Mean/Argmax for Eval)
        policy_out, _, h_next = model(obs_t, pa_t, pr_t, pd_t, h)
        
        if is_discrete:
            # policy_out: (N, action_dim)
            act_env = torch.argmax(policy_out, dim=-1).cpu().numpy()
        else:
            # policy_out: (N, action_dim * 2) -> first half is mu
            mu, _ = policy_out
            act_env = torch.tanh(mu).cpu().numpy()

        # Step environments
        next_obs, rew, term, trunc, _ = envs.step(act_env)
        done = np.logical_or(term, trunc)
        
        # --- Accumulation Logic ---
        # Only add rewards if this environment hasn't finished yet
        total_rewards += rew * (~env_finished)
        
        # Mark as finished if it just hit done
        env_finished = np.logical_or(env_finished, done)

        # Break early if every environment has completed at least one episode
        if env_finished.all():
            break

        # Transition
        obs = next_obs
        h = h_next
        prev_a = act_env
        prev_r = rew
        prev_done = done.astype(np.float32)

        # Clean-up RL2 inputs for next step
        for i in range(num_envs):
            if done[i]:
                if is_discrete: prev_a[i] = 0
                else: prev_a[i] = 0.0
                prev_r[i] = 0.0

    return float(np.mean(total_rewards))


