# RL²-Torch: Fast Adaptation through Meta-Reinforcement Learning

A PyTorch implementation of **RL² (Reinforcement Learning Squared)**, a meta-reinforcement learning algorithm that enables agents to quickly adapt to new tasks from minimal experience.

## 📖 What is RL² (Meta-Reinforcement Learning)?

**RL²** is a meta-learning approach that treats the learning algorithm itself as a recurrent neural network. Instead of explicitly programming an adaptation mechanism, the agent learns to adapt by encoding task-specific information in its recurrent hidden state.

### Key Insight

Traditional RL agents learn slowly through trial-and-error on each new task. **RL²** meta-trains an agent across a distribution of tasks, teaching it to:

1. **Recognize task structure** from observations, rewards, and actions
2. **Encode task-specific information** in GRU hidden states (acting as "fast weights")
3. **Adapt its policy rapidly** within a single episode using this encoded context

### How It Works

```
Episode 1 (Task A):
  Step 1: h₀=None → Explore → h₁ (uncertain)
  Step 2: h₁ → Explore more → h₂ (learning)
  Step K: hₖ → Exploit → Optimized for Task A

Episode 2 (New Task B):
  Step 1: h₀=None → Explore → h₁ (fast recognition)
  Step 5: h₅ → Already adapted to Task B!
```

The **recurrent hidden state** becomes a learned adaptation mechanism that generalizes across tasks.

### Meta-Training Process

1. **Sample a batch of tasks** (e.g., CartPole with different wind strengths)
2. **Roll out episodes** where the GRU learns to adapt within each episode
3. **Update the policy** using PPO to maximize cumulative return across all tasks
4. **Repeat** until the agent learns a general adaptation strategy

After meta-training, the agent can quickly adapt to **unseen tasks** from the same distribution with minimal experience.

---

## 🏗️ Repository Structure

```
RL2-Torch/
├── agents/                          # Core neural network architectures
│   ├── nn_tools.py                 # MLP builder with orthogonal initialization
│   ├── feature_extractor.py        # RL2GRUFeatureExtractor (context encoder)
│   ├── actor_critic.py             # RL2ActorCritic (policy + value heads)
│   └── __init__.py
├── utils/                           # Supporting utilities
│   ├── buffer.py                   # RL2RolloutBuffer (trajectory storage)
│   ├── distributions.py            # TanhNormal for bounded continuous actions
│   ├── normalizers.py              # RunningMeanStd input normalization
│   └── evaluation.py               # Meta-evaluation protocol
├── experiments/                     # Task-specific implementations
│   ├── cartpole_wind.py            # CartPole with dynamic wind (discrete)
│   ├── pendulum_wind.py            # Pendulum with wind torque (continuous)
│   └── halfcheetah_action.py       # HalfCheetah with action offset (continuous)
├── train.py                         # Main RL² + PPO training loop
├── checkpoints/                     # Saved model weights
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🧠 Neural Network Architecture

### Overview

The RL² agent consists of three main components:

1. **Feature Extractor** (RL2GRUFeatureExtractor): Encodes task context
2. **Actor Head**: Outputs policy distribution
3. **Critic Head**: Estimates state value

### 1. RL2GRUFeatureExtractor - Context Encoder

The **core innovation** of RL²: a recurrent encoder that processes the history of observations, actions, and rewards to build task-specific representations.

```
Input: [state, previous_action, previous_reward, done_signal]
    ↓
[Normalization & Embedding Layer]
    • State: RunningMeanStd normalization
    • Action (Discrete): Adaptive embedding (4-16 dims based on action space)
    • Action (Continuous): Raw values with optional normalization
    • Reward: RunningMeanStd normalization
    ↓
[Concatenate All Inputs]
    ↓
[Linear Encoder: combined_dim → h_dim]
    ↓
[LayerNorm + ReLU]
    ↓
[GRU Cell: h_dim → h_dim, single layer]
    ↓
Output: Hidden state h ∈ ℝ^(1 × batch × h_dim)
        (encodes task-specific context)
```

**Key Design Choices:**

- **GRU over LSTM:** Simpler architecture, fewer parameters, sufficient for adaptation
- **Adaptive Action Embedding:** `embed_dim = min(16, max(4, ⌈log₂(action_dim)⌉ × 2))`
- **Layer Normalization:** Stabilizes pre-GRU representations
- **Hidden State as Memory:** Carries learned task information across timesteps

**Why This Works:**
The GRU hidden state `h_t` evolves as:
```
h_t = GRU(encoder([s_t, a_{t-1}, r_{t-1}, done_{t-1}]), h_{t-1})
```
Over the course of an episode, `h` accumulates information about:
- Task-specific reward structure
- Environment dynamics (wind direction, friction, etc.)
- Optimal policy parameters for the current task

### 2. RL2ActorCritic - Policy and Value Heads

```
GRU Hidden State (h_dim)
    ↓
    ├──────────────────────┬──────────────────────┐
    ↓                      ↓
[Actor MLP]           [Critic MLP]
(h_dim → hidden → out) (h_dim → hidden → 1)
    ↓                      ↓
Policy Distribution    Value Estimate V(s)
```

**Discrete Actions (e.g., CartPole):**
- Output: `action_dim` logits
- Distribution: `Categorical(logits)`

**Continuous Actions (e.g., HalfCheetah, Pendulum):**
- Output: `2 × action_dim` parameters
- Split into: `[μ, log_std]` where `log_std ∈ [-20, 2]` (clamped for stability)
- Distribution:
  - **Unbounded:** `Normal(μ, std)`
  - **Bounded:** `TanhNormal(μ, std, low, high)` (uses change-of-variables)

**Design Pattern:**
- **Shared Feature Extraction:** Both heads use the same GRU-encoded features
- **Decoupled Optimization:** Separate MLPs allow independent policy/value learning

### 3. MLP (Multi-Layer Perceptron)

**Location:** `agents/nn_tools.py`

```python
def mlp(in_dim, out_dim, hidden_dims=(64, 64), activation=nn.Tanh, last_std=0.01)
```

- **Structure:** `Linear → Activation → ... → Linear` (no activation on final layer)
- **Initialization:**
  - Hidden layers: Orthogonal with `std = √2` (preserves gradient flow)
  - Output layer: Orthogonal with `std = 0.01` (stabilizes initial policy)
- **Activation:** Tanh (better than ReLU for PPO stability)

### 4. TanhNormal Distribution (Bounded Continuous Actions)

**Location:** `utils/distributions.py`

For continuous actions with bounds (e.g., `[-1, 1]`):

```python
u ~ Normal(μ, σ)                     # Sample from unbounded Gaussian
a = bias + scale × tanh(u)          # Squash to [low, high]

# Compute log probability with change-of-variables:
log p(a) = log p(u) - log(scale) - log(1 - tanh²(u))
```

**Why This Matters:**
- **Numerically Stable:** Avoids computing `arctanh(a)` which can be unstable
- **Stores Raw Samples:** Keeps `u` for stable log probability computation
- **Supports Gradient Flow:** Smooth differentiable transformation

---

## 🔄 Training Process (RL² + PPO)

### Phase 1: Experience Collection

```python
for update in range(total_updates):
    # Sample a batch of tasks (e.g., different wind magnitudes)
    tasks = sample_tasks_train(num_envs)
    envs = make_envs_train(tasks)

    h = None  # Initialize hidden state
    for step in range(horizon):
        # Forward pass through RL² agent
        policy, value, h_next = model(obs, prev_action, prev_reward, prev_done, h)

        # Sample action and execute in environment
        action = policy.sample()
        next_obs, reward, done, info = envs.step(action)

        # Store trajectory with hidden state
        buffer.store(obs, action, reward, done, value, log_prob, h)

        # Update state and hidden state
        obs, h = next_obs, h_next
        if done: h = None  # Reset hidden state for new episode
```

**Key Points:**
- Each task gets its own GRU trajectory (hidden state evolves independently)
- Hidden state `h` is reset at episode boundaries
- Within an episode, `h` accumulates task-specific information

### Phase 2: Advantage Computation

Uses **Generalized Advantage Estimation (GAE)**:

```python
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
A_t = Σ_{k=0}^∞ (γλ)^k × δ_{t+k}
```

Where:
- `γ = 0.99` (discount factor)
- `λ = 0.95` (GAE lambda, balances bias-variance)

### Phase 3: PPO Optimization

```python
for epoch in range(ppo_epochs):
    for batch in buffer.get_batches(minibatch_chunks, chunk_len):
        # Re-forward with stored hidden states
        new_policy, new_value = model(batch.obs, batch.prev_action,
                                       batch.prev_reward, batch.prev_done, batch.h)

        # Compute PPO clipped loss
        ratio = exp(new_log_prob - old_log_prob)
        clipped_ratio = clip(ratio, 1-ε, 1+ε)
        policy_loss = -mean(min(ratio × A, clipped_ratio × A))

        # Value loss (MSE)
        value_loss = 0.5 × mean((new_value - returns)²)

        # Entropy bonus (encourages exploration)
        entropy_loss = -0.01 × mean(policy.entropy())

        # Total loss
        loss = policy_loss + value_loss + entropy_loss

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
```

**Chunk-Based Training:**
- Trajectories are split into chunks of length `chunk_len` (default: 32)
- Each chunk maintains its initial hidden state from collection
- Enables efficient batched backpropagation through time (BPTT)

---

## 🎯 Implemented Environments

### 1. CartPole with Wind (Discrete)

**File:** `experiments/cartpole_wind.py`

- **State Dimension:** 4 (position, velocity, angle, angular velocity)
- **Action Space:** Discrete (2 actions: push left/right)
- **Task Distribution:** Wind force ∈ [-5, 5]
- **Hidden Dimension:** 64
- **Challenge:** Agent must infer wind direction from state trajectory and adjust policy

### 2. Pendulum with Wind (Continuous, Bounded)

**File:** `experiments/pendulum_wind.py`

- **State Dimension:** 3 (cos(θ), sin(θ), angular velocity)
- **Action Space:** Continuous, bounded ∈ [-2, 2] (torque)
- **Task Distribution:** Wind torque ∈ [-0.5, 0.5]
- **Hidden Dimension:** 256
- **Challenge:** Precise continuous control with unknown external torque

### 3. HalfCheetah with Action Offset (Continuous, Bounded)

**File:** `experiments/halfcheetah_action.py`

- **State Dimension:** 17 (joint positions and velocities)
- **Action Space:** Continuous, bounded ∈ [-1, 1]^6 (joint torques)
- **Task Distribution:** Action offset ∈ [-0.2, 0.2] per dimension
- **Hidden Dimension:** 256
- **Challenge:** High-dimensional continuous control with systematic action bias

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RL2-Torch
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies

- `torch>=2.10.0` - Deep learning framework
- `gymnasium>=1.2.3` - Standard RL environments (successor to OpenAI Gym)
- `numpy>=2.4.0` - Numerical computing
- `matplotlib>=3.10.0` - Plotting and visualization
- `mujoco>=3.4.0` - Physics simulator for HalfCheetah
- `metaworld>=3.0.0` - Meta-learning benchmark environments

---

## 🎮 Running Experiments

### Quick Start

Run any of the three experiments using Python module syntax:

```bash
# CartPole with dynamic wind (discrete actions)
python -m experiments.cartpole_wind

# Pendulum with wind torque (continuous bounded actions)
python -m experiments.pendulum_wind

# HalfCheetah with action offset (continuous bounded actions)
python -m experiments.halfcheetah_action
```

### What Happens During Training

Each experiment will:

1. **Initialize** the RL² agent with appropriate architecture
2. **Meta-train** across a distribution of tasks (e.g., different wind strengths)
3. **Print progress** every update:
   ```
   Update 0020 | Reward: 45.23 ± 12.45 | Value Loss: 0.532 | Policy Loss: -0.021
   ```
4. **Evaluate periodically** (every 20 updates by default):
   ```
   [Eval] Update 0020 | Mean Return: 78.45 ± 15.32 (10 held-out tasks)
   ```
5. **Save best model** to `checkpoints/` directory

### Training Output

During training, you'll see:

- **Reward:** Mean episode return across all training tasks
- **Value Loss:** Mean squared error of value function predictions
- **Policy Loss:** PPO clipped surrogate loss
- **Eval Returns:** Performance on held-out test tasks (measures generalization)

### Monitoring Adaptation

The agent's ability to adapt is measured by:

1. **Early episode performance:** How well does it perform in steps 1-100 of a new task?
2. **Late episode performance:** How much does performance improve by steps 100-200?
3. **Generalization gap:** Performance difference between training and held-out tasks

### Checkpoints

Best models are automatically saved to:
```
checkpoints/
├── best_cartpole_wind.pt
├── best_pendulum_wind.pt
└── best_halfcheetah_action.pt
```

Each checkpoint contains:
- Model state dict (`model.state_dict()`)
- Optimizer state
- Running normalization statistics
- Training metadata (update count, best reward)

---

## 📊 Expected Results

### CartPole with Wind

- **Training Tasks:** Wind ∈ {-5, -2.5, 0, 2.5, 5}
- **Test Tasks:** Wind ∈ {-3.75, -1.25, 1.25, 3.75}
- **Expected Performance:**
  - Random policy: ~20 steps
  - After meta-training: >150 steps on unseen winds
  - Adaptation speed: Near-optimal within 50 steps

### Pendulum with Wind

- **Training Tasks:** Wind torque ∈ [-0.5, 0.5] (uniform sampling)
- **Test Tasks:** Held-out wind torques
- **Expected Performance:**
  - Random policy: ~-1200 cumulative reward
  - After meta-training: >-400 on unseen winds
  - Adaptation speed: Identifies wind direction within 20 steps

### HalfCheetah with Action Offset

- **Training Tasks:** Action offset ∈ [-0.2, 0.2]^6 (uniform sampling)
- **Test Tasks:** Held-out offsets
- **Expected Performance:**
  - Random policy: ~-100 cumulative reward
  - After meta-training: >2000 on unseen offsets
  - Adaptation speed: Compensates for bias within 100 steps

---

## 🔧 Hyperparameter Configuration

### Default Hyperparameters (in `train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_envs` | 4 | Number of parallel environments (task diversity) |
| `total_updates` | 1000 | Total PPO update iterations |
| `horizon` | 1024 | Steps collected per update per environment |
| `chunk_len` | 32 | Sequence length for BPTT |
| `ppo_epochs` | 4 | Number of passes through collected data |
| `minibatch_chunks` | 16 | Number of minibatches per epoch |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda (bias-variance tradeoff) |
| `lr` | 3e-4 | Learning rate (Adam optimizer) |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `eval_interval` | 20 | Evaluate every N updates |
| `eval_tasks_count` | 10 | Number of held-out tasks for evaluation |

### Architecture Hyperparameters

| Environment | State Dim | Action Dim | Hidden Dim | Actor MLP | Critic MLP |
|-------------|-----------|------------|------------|-----------|------------|
| CartPole | 4 | 2 | 64 | (64, 64) | (64, 64) |
| Pendulum | 3 | 1 | 256 | (64, 64) | (64, 64) |
| HalfCheetah | 17 | 6 | 256 | (64, 64) | (64, 64) |

**Rule of Thumb:**
- Discrete actions → smaller hidden dim (64)
- Continuous actions → larger hidden dim (128-256)
- Higher state dimensions → larger hidden dim

---

## 🛡️ Stability Mechanisms

The implementation includes several techniques to ensure stable training:

1. **Gradient Clipping** (`max_grad_norm=0.5`)
   - Prevents exploding gradients in recurrent networks
   - Essential for GRU training stability

2. **Advantage Normalization**
   - Per-minibatch standardization: `A_norm = (A - μ) / σ`
   - Reduces variance in policy gradient estimates

3. **PPO Clipping** (`ε=0.2`)
   - Constrains policy updates to small steps
   - Prevents catastrophic policy collapse

4. **Entropy Regularization** (`coef=0.01`)
   - Encourages exploration during meta-training
   - Prevents premature convergence to suboptimal policies

5. **Layer Normalization**
   - Applied before GRU in feature extractor
   - Stabilizes internal representations

6. **Orthogonal Initialization**
   - Maintains gradient flow through deep networks
   - Small output layer std (0.01) stabilizes initial policy

7. **Input Normalization**
   - Running mean-std for states and rewards
   - Uses Welford's algorithm for numerical stability
   - Optional max_samples limit prevents distribution shift

8. **Log-Std Clamping** (for continuous actions)
   - Clamps `log_std ∈ [-20, 2]`
   - Prevents numerical overflow/underflow in Gaussian distributions

---

## 📚 Key Concepts and Design Patterns

### 1. Hidden State as Fast Weights

The GRU hidden state acts as a **learned parameter vector** that adapts within an episode:

```
Traditional RL: Fixed weights θ, slow updates via gradient descent
RL²:           Fast weights h_t (updated by GRU), slow weights θ (updated by PPO)
```

This enables **two-timescale learning**:
- **Outer loop (slow):** Meta-train θ to produce good adaptation behavior
- **Inner loop (fast):** Adapt h_t to specific task within episode

### 2. Dependency Injection

Feature extractor is injected into actor-critic:

```python
feat_extractor = RL2GRUFeatureExtractor(state_dim, action_dim, h_dim, ...)
model = RL2ActorCritic(feat_extractor, actor_mlp=(64, 64), critic_mlp=(64, 64))
```

**Benefits:**
- Single source of truth for dimensions (no manual matching)
- Flexible architecture composition
- Easy to swap different feature extractors

### 3. Modular Normalization

```python
feat_extractor = RL2GRUFeatureExtractor(
    state_norm=RunningMeanStd1DNormalizer(state_dim),
    reward_norm=RunningMeanStd1DNormalizer(1),
    action_norm=None  # Optional
)
```

**Benefits:**
- Mix-and-match normalization strategies
- Can disable normalization per-component
- Maintains running statistics across training

### 4. Chunk-Based BPTT

Trajectories are split into chunks for efficient training:

```python
# Collection: Store full trajectories with hidden states
for t in range(horizon):
    action, value, h_next = model(..., h)
    buffer.store(..., h)  # Store hidden state
    h = h_next

# Training: Split into chunks and backprop
chunks = buffer.build_chunks(chunk_len=32)
for chunk in chunks:
    # Each chunk starts with stored hidden state from collection
    policy, value = model(..., chunk.h_init)
    loss = ppo_loss(...)
    loss.backward()  # BPTT over chunk_len steps
```

**Benefits:**
- Balances computational efficiency and credit assignment
- Prevents vanishing gradients over long sequences
- Allows larger effective context than full BPTT

---

## 🧪 Extending the Framework

### Adding a New Task

1. **Create environment with task distribution:**

```python
# experiments/my_environment.py
import gymnasium as gym
from gymnasium import Wrapper

class MyEnvWithTask(Wrapper):
    def __init__(self, task_param):
        env = gym.make('MyEnv-v0')
        super().__init__(env)
        self.task_param = task_param

    def reset(self, **kwargs):
        # Apply task-specific modifications
        return self.env.reset(**kwargs)
```

2. **Define task sampling and environment creation:**

```python
def sample_tasks_train(num_tasks):
    return [{'task_param': random.uniform(-1, 1)} for _ in range(num_tasks)]

def make_envs_train(tasks):
    return SyncVectorEnv([
        lambda task=task: MyEnvWithTask(**task) for task in tasks
    ])
```

3. **Configure and train:**

```python
from agents.feature_extractor import RL2GRUFeatureExtractor
from agents.actor_critic import RL2ActorCritic
from train import train_rl2_ppo

def model_factory():
    feat_extractor = RL2GRUFeatureExtractor(
        state_dim=...,
        action_dim=...,
        h_dim=128,
        is_discrete=...,
    )
    return RL2ActorCritic(feat_extractor, actor_mlp=(64, 64), critic_mlp=(64, 64))

train_rl2_ppo(
    model_factory=model_factory,
    sample_tasks_train=sample_tasks_train,
    make_envs_train=make_envs_train,
    ...
)
```

### Modifying Architecture

**Change hidden dimension:**
```python
feat_extractor = RL2GRUFeatureExtractor(..., h_dim=512)  # Larger capacity
```

**Change actor/critic MLPs:**
```python
model = RL2ActorCritic(feat_extractor,
                       actor_mlp=(128, 128, 128),  # Deeper actor
                       critic_mlp=(64, 64))         # Keep critic shallow
```

**Add action normalization:**
```python
feat_extractor = RL2GRUFeatureExtractor(
    ...,
    action_norm=RunningMeanStd1DNormalizer(action_dim)
)
```

---

## 📖 References

### Original Papers

1. **RL² Paper:**
   - Duan, Y., Schulman, J., Chen, X., Bartlett, P. L., Sutskever, I., & Abbeel, P. (2016).
   - *RL²: Fast Reinforcement Learning via Slow Reinforcement Learning*
   - arXiv:1611.02779
   - [https://arxiv.org/abs/1611.02779](https://arxiv.org/abs/1611.02779)

2. **PPO Paper:**
   - Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
   - *Proximal Policy Optimization Algorithms*
   - arXiv:1707.06347
   - [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

3. **GAE Paper:**
   - Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
   - *High-Dimensional Continuous Control Using Generalized Advantage Estimation*
   - arXiv:1506.02438
   - [https://arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)

### Related Work

- **MAML (Model-Agnostic Meta-Learning):** Explicit meta-learning through gradient-based adaptation
- **Meta-World:** Benchmark suite for meta-reinforcement learning
- **Learning to Reinforcement Learn:** Neuroscience-inspired perspective on RL²

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add visualization tools for hidden state analysis
- [ ] Implement additional meta-RL baselines (MAML, PEARL)
- [ ] Support for more environments (Meta-World, ProcGen)
- [ ] Multi-layer GRU/LSTM variants
- [ ] Attention-based feature extractors
- [ ] Distributed training support
- [ ] Hyperparameter tuning utilities
- [ ] Pre-trained model zoo

---

## 📄 License

[Add your license here]

---

## 🙏 Acknowledgments

This implementation is based on the RL² paper by Duan et al. (2016) and uses:
- PyTorch for deep learning
- Gymnasium for RL environments
- MuJoCo for physics simulation
- PPO algorithm for policy optimization

---

## 📧 Contact

[Add your contact information here]

---

**Happy Meta-Learning!** 🚀
