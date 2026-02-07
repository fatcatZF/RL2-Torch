import gymnasium as gym 
from gymnasium import Wrapper
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.vector import SyncVectorEnv
from gymnasium import logger


import numpy as np 


from typing import List, Dict, Any 
import random

from agents.actor_critic import RL2ActorCritic
from agents.feature_extractor import RL2GRUFeatureExtractor
from utils.normalizers import RunningMeanStd1DNormalizer


from train import train_rl2_ppo

import os 

# Environment

class CartPoleEnvWithWind(CartPoleEnv):
    def __init__(self, force_mag=10.0, wind_mag=0, **kwargs):
        super().__init__(**kwargs)
        self.force_mag = force_mag
        assert wind_mag >= -5 and wind_mag <=5, "The wind mag should within [-5, 5]"
        self.wind_mag = wind_mag

    def step(self, action):
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        #force = force + self.wind_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta + self.wind_mag
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
    





gym.register(
    'CartPoleWithWind-v1',
    CartPoleEnvWithWind,
    max_episode_steps=500
)


class RewardShapingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Modify the reward based on observation
        reward = self.modify_reward(observation, reward)
        return observation, reward, terminated, truncated, info

    def modify_reward(self, observation, reward):
        # Define custom reward modification logic
        # For example, add a penalty if the pole angle is too high
        pole_angle = observation[2]
        reward = np.cos(pole_angle)
        return reward




def sample_tasks_train(num_envs: int = 4) -> List[Dict[str, Any]]:
    """
    Generates a list of task configurations for CartPole with random wind.
    Each task is a dictionary that can be passed as kwargs to the environment.
    """
    tasks = []
    for _ in range(num_envs):
        # Sample wind_mag uniformly between -5 and 5
        wind = random.uniform(-5.0, 5.0)
        tasks.append({
            "wind_mag": wind
            # You can also add "force_mag" here if you want to meta-learn that too
        })
    return tasks





def sample_tasks_eval(num_envs: int = 16) -> List[Dict[str, Any]]:
    """
    Generates a list of task configurations with wind_mag evenly spaced 
    between -5 and 5.
    """
    # Create num_envs points evenly spaced between -5 and 5
    # np.linspace is ideal for ensuring the boundaries are included
    wind_values = np.linspace(-5.0, 5.0, num=num_envs)
    
    tasks = []
    for wind in wind_values:
        tasks.append({
            "wind_mag": float(wind)
        })
    return tasks


def make_envs(task_configs: List[Dict[str, Any]]):
    def make_env(config):
        return lambda: RewardShapingWrapper(gym.make("CartPoleWithWind-v1", **config))
    return SyncVectorEnv([make_env(cfg) for cfg in task_configs])







def cartpole_model_factory(h_dim=64):
    """
    Creates an RL² model specifically for CartPole with Wind.
    """
    # 1. Define dimensions for CartPole-v1
    state_dim = 4 
    action_dim = 2 # Left, Right
    
    # 2. Setup Normalizers (Highly recommended for PPO stability)
    state_norm = RunningMeanStd1DNormalizer(shape=(state_dim,), max_samples=5000)
    reward_norm = RunningMeanStd1DNormalizer(shape=(1,), max_samples=5000)
    
    # 3. Build the Feature Extractor
    # RL² needs to see state, previous action, previous reward, and done
    feature_extractor = RL2GRUFeatureExtractor(
        state_dim=state_dim,
        action_dim=action_dim,
        h_dim=h_dim,
        is_discrete=True,
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
    save_path = os.path.join(project_root, "checkpoints", "cartpole_wind.pt")

    trained_model = train_rl2_ppo(
        model_factory=lambda: cartpole_model_factory(h_dim=64),
        sample_tasks_train=sample_tasks_train,
        make_envs_train=make_envs, # In your file, make_envs handles the dict mapping
        sample_tasks_eval=sample_tasks_eval,
        make_envs_eval=make_envs,
        action_dim=2,
        h_dim=64,
        is_discrete=True,
        num_envs=4,
        save_path=save_path,
        )









