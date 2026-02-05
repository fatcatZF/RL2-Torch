import torch 
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import math 






class TanhNormal:
    """
    A Tanh-transformed Normal distribution.
    By storing 'u' (raw samples), we can avoid numerical instability of atanh.
    """
    def __init__(self, mu: torch.Tensor, std: torch.Tensor,
                 low: torch.Tensor | float | None = None,
                 high: torch.Tensor | float | None = None):
        self.device = mu.device
        self.dtype = mu.dtype
        self.base = Normal(mu, std)

        if low is None or high is None:
            self._scale, self._bias = 1.0, 0.0
        else:
            low_t = torch.as_tensor(low, device=self.device, dtype=self.dtype)
            high_t = torch.as_tensor(high, device=self.device, dtype=self.dtype)
            self._scale = (high_t - low_t) / 2.0
            self._bias = (high_t + low_t) / 2.0

    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        return self._bias + self._scale * torch.tanh(u)

    def _unsquash(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Map [low, high] -> [-1, 1]
        z = (a - self._bias) / (self._scale + 1e-8)
        # Numerical safety: clip to avoid infinity at +/- 1.0
        z = torch.clamp(z, -1.0 + 1e-7, 1.0 - 1e-7)
        u = torch.atanh(z)
        return u, z

    def rsample(self, sample_shape=torch.Size()) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (squashed_action, raw_u)"""
        u = self.base.rsample(sample_shape)
        return self._squash(u), u

    def sample(self, sample_shape=torch.Size()) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (squashed_action, raw_u)"""
        u = self.base.sample(sample_shape)
        return self._squash(u), u

    def log_prob_from_u(self, u: torch.Tensor) -> torch.Tensor:
        """
        Calculates log_prob directly from the Gaussian sample u.
        This is much more numerically stable than calculating from 'a'.
        """
        logp_u = self.base.log_prob(u)
        
        # Change of variables correction: log(da/du)
        # a = bias + scale * tanh(u) -> da/du = scale * (1 - tanh^2(u))
        # log(da/du) = log(scale) + log(1 - tanh^2(u))
        
        # Stable log(1 - tanh^2(u))
        tanh_det = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        
        return logp_u - torch.log(torch.as_tensor(self._scale + 1e-8, device=self.device)) - tanh_det

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        """Standard log_prob, uses unsquash (less stable than log_prob_from_u)"""
        u, _ = self._unsquash(a)
        return self.log_prob_from_u(u)

    def entropy(self) -> torch.Tensor:
        return self.base.entropy()

    @property
    def mean(self):
        return self._squash(self.base.mean)

    @property
    def stddev(self):
        return self.base.stddev
    



def make_action_dist(policy_out, is_discrete: bool, 
                     action_low=None, action_high=None):
    """
    Factory function to build a torch distribution from model output.
    """
    if is_discrete:
        # Categorical expects logits (unnormalized log probabilities)
        return Categorical(logits=policy_out)
    else:
        mu, log_std = policy_out
        std = torch.exp(log_std)
        
        # If bounds are provided, use our stable TanhNormal
        if action_low is not None and action_high is not None:
            return TanhNormal(mu, std, action_low, action_high)
        
        # Otherwise, fall back to a standard Unbounded Normal
        return Normal(mu, std)