import torch 
import torch.nn as nn 
import numpy as np 
from typing import Any



@torch.no_grad()
def run_meta_eval(
    model: nn.Module,
    envs: Any, 
    device: torch.device,
    is_discrete: bool,
    max_steps: int, 
    action_dim: int
) -> float:
    model.eval()
    num_envs = envs.num_envs
    obs, _ = envs.reset()
    
    # RL² Context initialization
    h = None
    if is_discrete:
        prev_a = np.zeros(num_envs, dtype=np.int64)
    else:
        prev_a = np.zeros((num_envs, action_dim), dtype=np.float32)
    
    prev_r = np.zeros(num_envs, dtype=np.float32)
    prev_done = np.ones(num_envs, dtype=np.float32)
    
    total_rewards = np.zeros(num_envs)
    env_finished = np.zeros(num_envs, dtype=bool)

    for t in range(max_steps):
        # 1. Convert to 3D Tensors (L=1, N=num_envs, D=dim)
        # Explicitly cast to float32 to avoid "Double vs Float" errors
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        pa_t = torch.as_tensor(prev_a, device=device).unsqueeze(0)
        pr_t = torch.as_tensor(prev_r, device=device, dtype=torch.float32).view(1, num_envs, 1)
        pd_t = torch.as_tensor(prev_done, device=device, dtype=torch.float32).view(1, num_envs, 1)

        # 2. Forward pass returns (L, N, D)
        policy_out, _, h_next = model(obs_t, pa_t, pr_t, pd_t, h)
        
        # Squeeze sequence dimension for action selection (1, N, D) -> (N, D)
        #policy_out = policy_out.squeeze(0)

        if is_discrete:
            # Select best action (Argmax) for evaluation
            logits = policy_out.squeeze(0)
            act_env = torch.argmax(logits, dim=-1).cpu().numpy()
        else:
            # For continuous, policy_out contains (mu, log_std)
            mu, _ = policy_out 
            mu = mu.squeeze(0)
            act_env = torch.tanh(mu).cpu().numpy()

        # 3. Environment Step
        next_obs, rew, term, trunc, _ = envs.step(act_env)
        done = np.logical_or(term, trunc)
        
        # Accumulate rewards only for active episodes
        total_rewards += rew * (~env_finished)
        env_finished = np.logical_or(env_finished, done)

        if env_finished.all():
            break

        # 4. Update Context for next step
        obs = next_obs
        h = h_next
        prev_a = act_env
        prev_r = rew
        prev_done = done.astype(np.float32)

        # Reset context inputs for finished environments within the meta-episode
        for i in range(num_envs):
            if done[i]:
                prev_a[i] = 0 if is_discrete else 0.0
                prev_r[i] = 0.0

    return float(np.mean(total_rewards))



@torch.no_grad()
def run_meta_eval_continuous():
    pass 