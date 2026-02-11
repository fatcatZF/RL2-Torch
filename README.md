# RL2-Torch

A PyTorch implementation of **RL²** (RL-Squared): Fast Reinforcement Learning via Slow Reinforcement Learning.

## Overview

This repository implements the RL² meta-reinforcement learning algorithm introduced in the paper ["RL²: Fast Reinforcement Learning via Slow Reinforcement Learning" (Duan et al., 2016)](https://arxiv.org/abs/1611.02779). RL² is a pioneering approach in meta-reinforcement learning that learns to learn - training an agent that can rapidly adapt to new tasks from the same distribution.

### What is RL²?

RL² addresses the fundamental challenge in reinforcement learning: while deep RL can learn sophisticated behaviors, it typically requires an enormous number of trials. In contrast, animals can learn new tasks in just a few trials by leveraging prior knowledge. RL² bridges this gap by:

- **Meta-Learning Approach**: Instead of designing a fast RL algorithm manually, RL² represents the learning algorithm itself as a recurrent neural network (RNN)
- **Learning to Learn**: The RNN's weights encode a general-purpose RL algorithm, learned through standard ("slow") RL training
- **Fast Adaptation**: Once trained, the agent can quickly adapt to new MDPs from the same distribution using only its hidden state

### Key Concept

The algorithm operates on two levels:
- **Outer Loop (Meta-Learning)**: Train across a distribution of MDPs to learn a general learning strategy
- **Inner Loop (Fast Adaptation)**: Use the learned RNN to quickly solve new tasks from the same distribution

The RNN receives observations, actions, rewards, and termination flags, maintaining its hidden state across episodes within a trial. This allows it to accumulate experience and adapt its behavior, effectively implementing a learned RL algorithm.

## Features

- PyTorch implementation of RL² algorithm
- Support for meta-learning across task distributions
- Recurrent policy architecture (GRU/LSTM)
- Training on various environments
- Checkpoint system for saving and loading trained models
- Experiment tracking and configuration

## Repository Structure

```
RL2-Torch/
├── agents/           # Agent implementations with recurrent architectures
├── checkpoints/      # Saved model checkpoints
├── experiments/      # Experiment configurations and results
├── utils/            # Utility functions and helper modules
├── train.py          # Main training script
├── requirements.txt  # Python dependencies
└── pyproject.toml    # Project configuration
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/fatcatZF/RL2-Torch.git
cd RL2-Torch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

## Usage

### Training

Run the main training script:
```bash
python train.py
```

The training process will:
1. Sample multiple MDPs from the task distribution
2. Train the meta-agent across these MDPs using PPO/TRPO
3. Save checkpoints periodically
4. Log training metrics and performance

### Configuration

Experiment configurations can be customized through:
- Command-line arguments
- Configuration files in the `experiments/` directory
- Modifying hyperparameters in the training script

### Key Hyperparameters

- **Number of episodes per trial (n)**: How many episodes the agent experiences in each MDP before moving to a new one
- **Meta-batch size**: Number of different MDPs sampled per meta-training iteration
- **RNN hidden size**: Capacity of the recurrent network's hidden state
- **Learning rate**: Step size for policy optimization
- **Episode length**: Maximum steps per episode

## Algorithm Details

### Architecture

The RL² agent consists of:
1. **Input Processing**: Encodes `(state, action, reward, done)` tuples into feature vectors
2. **Recurrent Core**: GRU/LSTM network that maintains hidden state across episodes
3. **Policy Head**: Outputs action distribution based on RNN hidden state
4. **Value Head**: Estimates state value for training

### Training Process

1. **Meta-Episode Generation**: For each meta-training iteration:
   - Sample N different MDPs from the task distribution
   - For each MDP, run multiple episodes maintaining RNN state
   - Collect trajectories including states, actions, rewards

2. **Policy Optimization**: 
   - Use PPO or TRPO to update the policy
   - Gradients backpropagate through the entire trial sequence
   - Update RNN weights to improve meta-learning performance

3. **Evaluation**:
   - Test on held-out MDPs from the same distribution
   - Measure adaptation speed and final performance
   - Compare against baseline algorithms

## Supported Environments

- Multi-armed bandits
- Tabular MDPs
- Continuous control tasks (MuJoCo)
- Navigation tasks
- Custom environment distributions

## Results

After training, the RL² agent demonstrates:
- Rapid adaptation to new tasks (few-shot learning)
- Performance comparable to specialized algorithms with optimality guarantees
- Generalization across task distributions
- Emergent exploration-exploitation strategies

## Citation

If you use this implementation in your research, please cite the original RL² paper:

```bibtex
@article{duan2016rl,
  title={RL²: Fast reinforcement learning via slow reinforcement learning},
  author={Duan, Yan and Schulman, John and Chen, Xi and Bartlett, Peter L and Sutskever, Ilya and Abbeel, Pieter},
  journal={arXiv preprint arXiv:1611.02779},
  year={2016}
}
```

## Related Work

- **Learning to Reinforcement Learn** (Wang et al., 2016): Concurrent work on meta-RL with similar ideas
- **MAML** (Finn et al., 2017): Model-Agnostic Meta-Learning for fast adaptation
- **SNAIL** (Mishra et al., 2017): Simple Neural Attentive Meta-Learner
- **Meta-RL Survey** (Beck et al., 2023): Comprehensive overview of meta-reinforcement learning

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original RL² paper by Duan et al.
- OpenAI and UC Berkeley research teams
- PyTorch community

## Contact

For questions or discussions about this implementation, please:
- Open an issue on GitHub
- Contact the repository maintainer

---

**Note**: This is an independent implementation for research and educational purposes. Performance may vary from the original paper depending on hyperparameters and environment setup.
