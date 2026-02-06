import os 
import torch 
import torch.nn as nn 
import numpy as np 
import random

from typing import Callable, Any, Optional, List, Dict
from utils.buffer import RL2RolloutBuffer, combine_chunks
from utils.distributions import make_action_dist




def train_rl2_ppo(
    # Model & Env Factories
    model_factory: Callable[[], nn.Module],
    sample_tasks_train: Callable[[int], List[Dict[str, Any]]],
    make_envs_train: Callable[[List[Dict[str, Any]]], Any],
    sample_tasks_eval: Callable[[int], List[Dict[str, Any]]],
    make_envs_eval: Callable[[List[Dict[str, Any]]], Any],
    # Dimensions & Config
    obs_dim: int,
    action_dim: int,
    h_dim: int = 64,
    is_discrete: bool = True,
    action_low=None, action_high=None,
    # Hyperparameters
    num_envs: int = 4,
    total_updates: int = 1000, 
    horizon: int = 1024,
    chunk_len: int = 32, 
    ppo_epochs: int = 4,
    minibatch_chunks: int = 16,
    gamma: float = 0.99, 
    lam: float = 0.95,
    lr: float = 3e-4,
    max_grad_norm: float = 0.5,
    # Evaluation
    eval_interval: int = 20,
    eval_tasks_count: int = 16,
    seed: int = 0,
    device: Optional[torch.device] = None,
    save_path: str = "best_rl2_model.pt"
):
    # --- Setup ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = RL2RolloutBuffer(num_envs=num_envs, device=device, is_discrete=is_discrete)
    
    best_eval_ret = -float('inf')

    for update in range(1, total_updates + 1):
        # 1. Sample and create training environments
        train_configs = sample_tasks_train(num_envs)
        envs = make_envs_train(train_configs)
        
        # --- Collection Phase ---
        model.eval()
        buffer.reset()
        obs, _ = envs.reset()
        
        # RL² context initialization
        h = None 
        prev_a = np.zeros(num_envs) if is_discrete else np.zeros((num_envs, action_dim))
        prev_r = np.zeros(num_envs)
        prev_done = np.ones(num_envs) # Start of new meta-episode

        for t in range(horizon):
            # Convert to tensors
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
            pa_t = torch.as_tensor(prev_a, device=device)
            pr_t = torch.as_tensor(prev_r, device=device).float().view(num_envs, 1)
            pd_t = torch.as_tensor(prev_done, device=device).float().view(num_envs, 1)

            with torch.no_grad():
                # Forward through GRU
                policy_out, value, h_next = model(obs_t, pa_t, pr_t, pd_t, h)
                
                # Distribution handles continuous/discrete logic
                dist = make_action_dist(policy_out, is_discrete, action_low, action_high)
                
                if is_discrete:
                    act_t = dist.sample() # (num_envs,)
                    logp_t = dist.log_prob(act_t)
                    act_env = act_t.cpu().numpy()
                    raw_act_store = act_env
                else:
                    # Stability: Sample raw Gaussian sample 'u'
                    raw_act_t = dist.sample_raw() 
                    act_t = torch.tanh(raw_act_t)
                    logp_t = dist.log_prob_from_raw(raw_act_t)
                    act_env = act_t.cpu().numpy()
                    raw_act_store = raw_act_t.cpu().numpy()

            next_obs, rew, term, trunc, _ = envs.step(act_env)
            done = np.logical_or(term, trunc).astype(np.float32)

            buffer.add(
                obs=obs, prev_a=prev_a, prev_r=prev_r, prev_done=prev_done,
                act=act_env, raw_act=raw_act_store, logp=logp_t.cpu().numpy(),
                val=value.squeeze(-1).cpu().numpy(), rew=rew, done=done, h=h
            )

            # Transition
            obs, h, prev_a, prev_r, prev_done = next_obs, h_next, act_env, rew, done

        # --- Post-Collection ---
        with torch.no_grad():
            # Final bootstrap value
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
            pa_t = torch.as_tensor(prev_a, device=device)
            pr_t = torch.as_tensor(prev_r, device=device).float().view(num_envs, 1)
            pd_t = torch.as_tensor(prev_done, device=device).float().view(num_envs, 1)
            _, last_v, _ = model(obs_t, pa_t, pr_t, pd_t, h)
        
        buffer.compute_gae(last_v.squeeze(-1), gamma, lam)
        chunks = buffer.build_chunks(chunk_len)
        envs.close()

        # --- Optimization Phase ---
        model.train()
        for _ in range(ppo_epochs):
            random.shuffle(chunks)
            for i in range(0, len(chunks), minibatch_chunks):
                mb = chunks[i:i + minibatch_chunks]
                b_obs, b_pa, b_pr, b_pd, b_act, b_raw_act, b_lp, b_adv, b_ret, b_h0, _ = combine_chunks(mb)

                # Re-run sequences
                p_out, val, _ = model(b_obs, b_pa, b_pr, b_pd, b_h0)
                new_dist = make_action_dist(p_out, is_discrete, action_low, action_high)

                # PPO Loss
                new_lp = new_dist.log_prob_from_raw(b_raw_act) if not is_discrete else new_dist.log_prob(b_act)
                ratio = torch.exp(new_lp - b_lp)
                # Normalize advantages per minibatch for stability
                adv_std = b_adv.std() + 1e-8
                norm_adv = (b_adv - b_adv.mean()) / adv_std
                
                surr1 = ratio * norm_adv
                surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * norm_adv # clip_eps=0.2

                pi_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * nn.MSELoss()(val.squeeze(-1), b_ret)
                ent_loss = new_dist.entropy().mean()

                loss = pi_loss + vf_loss - 0.01 * ent_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # --- Evaluation & Saving ---
        if update % eval_interval == 0:
            eval_configs = sample_tasks_eval(eval_tasks_count)
            eval_envs = make_envs_eval(eval_configs)
            avg_return = run_meta_eval(model, eval_envs, device, is_discrete, horizon, action_dim)
            eval_envs.close()

            print(f"Update {update:4d} | Meta-Eval Return: {avg_return:8.2f}")
            if avg_return > best_eval_ret:
                best_eval_ret = avg_return
                torch.save(model.state_dict(), save_path)
                print(f"--> Saved best model (Return: {best_eval_ret:.2f})")

    # Return best model
    model.load_state_dict(torch.load(save_path))
    return model

def run_meta_eval(model, envs, device, is_discrete, horizon, action_dim):
    """Simple evaluation loop across vectorized meta-tasks."""
    model.eval()
    n = envs.num_envs
    obs, _ = envs.reset()
    h = None
    pa = np.zeros(n) if is_discrete else np.zeros((n, action_dim))
    pr, pd = np.zeros(n), np.ones(n)
    total_rew = np.zeros(n)

    for _ in range(horizon):
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        pa_t = torch.as_tensor(pa, device=device)
        pr_t = torch.as_tensor(pr, device=device).float().view(n, 1)
        pd_t = torch.as_tensor(pd, device=device).float().view(n, 1)

        with torch.no_grad():
            p_out, _, h_next = model(obs_t, pa_t, pr_t, pd_t, h)
            dist = make_action_dist(p_out, is_discrete)
            act = dist.sample().cpu().numpy() if is_discrete else torch.tanh(dist.sample()).cpu().numpy()
        
        obs, rew, term, trunc, _ = envs.step(act)
        total_rew += rew
        h, pa, pr, pd = h_next, act, rew, np.logical_or(term, trunc).astype(np.float32)
        
    return total_rew.mean()