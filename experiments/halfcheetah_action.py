import gymnasium as gym 
from gymnasium.wrappers import TransformReward, TransformAction, TransformObservation
from gymnasium.vector import SyncVectorEnv
from gymnasium.spaces import Box


import numpy as np 


from agents.actor_critic import RL2ActorCritic
from agents.feature_extractor import RL2GRUFeatureExtractor
from utils.normalizers import RunningMeanStd1DNormalizer


from train import train_rl2_ppo

import os 
import random

from typing import List, Dict, Any 








def sample_tasks_train(num_envs: int = 4) -> List[Dict[str, Any]]:
    """
    sample action modifier
    """
    tasks = []
    for _ in range(num_envs):
        #sample action modifier
        a_m = np.random.uniform(low=-0.2, high=0.2) 
        tasks.append({
            "a_m": a_m
        })
    return tasks




def sample_tasks_eval(num_envs: int = 16) -> List[Dict[str, Any]]:
    """
    sample action modifier
    """
    a_ms = np.linspace(-0.2, 0.2, num=num_envs)
    
    tasks = []
    for a_m in a_ms:
        tasks.append({
            "a_m": float(a_m)
        })
    return tasks




def make_envs(task_configs: List[Dict[str, Any]]):
    def make_env(config):
        a_m = config["a_m"]
        return lambda: TransformAction(gym.make("HalfCheetah-v5"),
                                                lambda a: a+a_m,
                                                Box(-1.0, 1.0, shape=(6,), dtype=np.float32))
    return SyncVectorEnv([make_env(cfg) for cfg in task_configs])





def halfcheetah_model_factory(h_dim=256):
    """
    Creates an RL² model specifically for halfcheetah.
    """
    # 1. Define dimensions for halfcheetah
    state_dim = 17 
    action_dim = 6 # Left, Right
    
    # 2. Setup Normalizers (Highly recommended for PPO stability)
    state_norm = RunningMeanStd1DNormalizer(shape=(state_dim,), max_samples=5000)
    reward_norm = RunningMeanStd1DNormalizer(shape=(1,), max_samples=5000)
    
    # 3. Build the Feature Extractor
    # RL² needs to see state, previous action, previous reward, and done
    feature_extractor = RL2GRUFeatureExtractor(
        state_dim=state_dim,
        action_dim=action_dim,
        h_dim=h_dim,
        is_discrete=False,
        state_norm=state_norm,
        reward_norm=reward_norm
    )
    
    # 4. Build the Actor-Critic model
    model = RL2ActorCritic(
        feat_extractor=feature_extractor,
        actor_mlp=(64, 64),
        critic_mlp=(64, 64)
    )
    
    return model




if __name__=="__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_path = os.path.join(project_root, "checkpoints", "halfcheetah.pt")

    trained_model = train_rl2_ppo(
        model_factory=lambda: halfcheetah_model_factory(h_dim=256),
        sample_tasks_train=sample_tasks_train,
        make_envs_train=make_envs, # In your file, make_envs handles the dict mapping
        sample_tasks_eval=sample_tasks_eval,
        make_envs_eval=make_envs,
        action_dim=6,
        h_dim=256,
        is_discrete=False,
        num_envs=4,
        save_path=save_path,
        horizon = 2000,
        )