import torch
import torch.nn as nn 



class RunningMeanStd1DNormalizer(nn.Module):
    def __init__(self, shape, eps: float=1e-6, 
                 max_samples: int = None):
        super().__init__()
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.eps = eps 
        self.max_samples = max_samples

        self.register_buffer("mean", torch.zeros(self.shape))
        self.register_buffer("var", torch.ones(self.shape))
        self.register_buffer("count", torch.full((), eps))
        # Track the total number of samples processed
        self.register_buffer("total_samples_seen", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        # check the update limit
        if self.max_samples is not None and self.total_samples_seen >= self.max_samples:
            return 
        
        x_flat = x.reshape(-1, x.shape[-1]) 
        batch_count = x_flat.shape[0]
        if batch_count == 0:
            return 
        
        batch_mean = torch.mean(x_flat, dim=0) 
        batch_var = torch.var(x_flat, dim=0, unbiased=False)
        delta = batch_mean - self.mean 
        total_count = self.count + batch_count

        # Welford's algorithm 
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / total_count)

        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(M2 / total_count, min=self.eps))
        self.count.copy_(total_count)
        self.total_samples_seen += batch_count

    
    def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        if self.training and update:
            self.update(x)

        return (x - self.mean) / torch.sqrt(self.var + self.eps)
    




class FixedMinMaxNormalizer(nn.Module):
    """
    Normalizes features to a specific range (e.g., [0, 1] or [-1, 1])
    using predefined physical limits.
    """
    def __init__(self, low, high, clip:  bool=True, target_range: tuple=(0, 1)):
        super().__init__() 
        # Convert to tensors and register as buffers
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))

        self.clip = clip 
        self.target_low, self.target_high = target_range 

    
    def forward(self, x: torch.Tensor, update: bool=False) -> torch.Tensor:
        # 'update' argument is ignored but kept for interface consistency
        if self.clip:
            x = torch.max(torch.min(x, self.high), self.low)
        
        # Standard MinMax formula: (x-low)/(high-low) maps x to [0, 1]
        x_norm = (x - self.low) / (self.high - self.low + 1e-6)

        # Map from [0, 1] to [target_low, target_high]
        if (self.target_low, self.target_high) != (0, 1):
            x_norm = x_norm * (self.target_high - self.target_low) + self.target_low

        return x_norm