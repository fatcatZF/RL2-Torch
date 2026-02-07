import gymnasium as gym 
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize


import numpy as np 

from typing import List, Dict, Any 
import random

from agents.actor_critic import RL2ActorCritic
from agents.feature_extractor import RL2GRUFeatureExtractor
from utils.normalizers import RunningMeanStd1DNormalizer


from train import train_rl2_ppo

import os 

# Environment
class PendlumEnvWithWind(PendulumEnv):
    def __init__(self, wind_torque=0.0):
        super().__init__()
        assert wind_torque >= -0.5 and wind_torque <= 0.5, "Wind torque must be between -0.5 and 0.5"
        self.wind_torque = wind_torque

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        total_torque = u + self.wind_torque

        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * total_torque) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, {}


gym.register(
    'PendulumWithWind-v1',
    PendlumEnvWithWind,
    max_episode_steps=200
)

def sample_tasks_train(num_tasks: int) -> List[Dict[str, Any]]:
    """
    Samples wind torque values uniformly between -0.5 and 0.5 for training.
    """
    return [{"wind_torque": np.random.uniform(-0.5, 0.5)} for _ in range(num_tasks)]

def sample_tasks_eval(num_tasks: int) -> List[Dict[str, Any]]:
    """
    Samples wind torque values for evaluation. 
    You can use uniform sampling or a fixed grid for more consistent benchmarking.
    """
    # Option A: Uniform sampling
    # return [{"wind_torque": np.random.uniform(-0.5, 0.5)} for _ in range(num_tasks)]
    
    # Option B: Fixed grid for reproducible evaluation
    torques = np.linspace(-0.5, 0.5, num_tasks)
    return [{"wind_torque": t} for t in torques]




def make_envs(tasks: List[Dict[str, Any]]):
    """
    Creates a vectorized gymnasium environment where each sub-env has a different wind torque.
    """
    def make_env(wind_torque):
        def _thunk():
            # Create the registered environment with the specific wind_torque
            env = gym.make('PendulumWithWind-v1', wind_torque=wind_torque)
            return env
        return _thunk

    # Create a list of thunks for each task configuration
    env_fns = [make_env(task["wind_torque"]) for task in tasks]
    
    # Wrap them in a SyncVectorEnv or AsyncVectorEnv
    return gym.vector.SyncVectorEnv(env_fns)





def pendulum_model_factory(h_dim=256):
    """
    Create RL2 model specificially for Pendulum with Wind.
    """
    state_dim = 3
    action_dim = 1

    state_norm = RunningMeanStd1DNormalizer(shape=(state_dim,), max_samples=1000)
    reward_norm = RunningMeanStd1DNormalizer(shape=(1,), max_samples=1000)
 
    feature_extractor = RL2GRUFeatureExtractor(
        state_dim=state_dim,
        action_dim=action_dim,
        h_dim=h_dim,
        is_discrete=False,
        state_norm=state_norm,
        reward_norm=reward_norm
    )

    model = RL2ActorCritic(
        feat_extractor=feature_extractor,
        actor_mlp=(64, 64),
        critic_mlp=(64, 64),
    )

    return model 




if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_path = os.path.join(project_root, "checkpoints", "pendulum_wind.pt")

    trained_model = train_rl2_ppo(
        model_factory=lambda: pendulum_model_factory(h_dim=256),
        sample_tasks_train=sample_tasks_train,
        make_envs_train=make_envs, # In your file, make_envs handles the dict mapping
        sample_tasks_eval=sample_tasks_eval,
        make_envs_eval=make_envs,
        action_dim=1,
        h_dim=256,
        is_discrete=False,
        action_low = -2.0,
        action_high = 2.0,
        num_envs=16,
        total_updates = 2000,
        save_path=save_path,
        )

