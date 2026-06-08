# Non-Stationary Structural Causal Bandits (NS-SCM-MAB)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-red)](https://openreview.net/pdf?id=F4LhOqhxkk)
[![Conference](https://img.shields.io/badge/Poster-NeurIPS%202025-blue)](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/119070.png?t=1763529959.1211314)
[![Blog Post](https://img.shields.io/badge/Blog-Explainer-green)](https://yeahoon-k.github.io/ns-scb-explainer/)

- **Authors:** Yeahoon Kwon, Yesong Choe, Soungmin Park, Neil Dhir, Sanghack Lee
- **Conference:** Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS 2025)
- **Poster Session:** Thursday, December 4, 2025 • 11:00 AM – 2:00 PM PST • San Diego, Exhibit Hall C,D,E #2603
- **Blog Post:** [Interactive Explainer](https://yeahoon-k.github.io/ns-scb-explainer/)

## Overview

**NS-SCMMAB** is a Python library for non-stationary structural causal bandits that addresses sequential decision-making in environments with evolving causal mechanisms. Unlike traditional multi-armed bandit (MAB) formulations that assume fixed reward distributions, our framework models how causal structures change over time and how interventions propagate temporally.

### Key Features

- **Temporal Causal Modeling**: Explicitly models how causal structures evolve over time using temporal structural causal models (SCMs)
- **Non-Myopic Intervention Strategies**: Identifies intervention sequences that maximize both immediate and long-term rewards through POMIS+ (Possibly-Optimal Minimal Intervention Sets with Future Support)
- **Graphical Causal Tools**: Provides graphical characterization and algorithms for identifying optimal intervention strategies in non-stationary environments
- **Comprehensive Experiments**: Includes all code to reproduce the experimental results from the NeurIPS 2025 paper

## Installation

### Requirements

- Python 3.9 or higher
- Operating System: Linux or macOS (tested on both)

### Installing from Source

The easiest way to install NS-SCMMAB is using pip:

```bash
git clone https://github.com/yeahoon-k/NS-SCMMAB.git
cd NS-SCMMAB
pip install -e .
```

This will automatically install all required dependencies:
- `numpy >= 1.21.2`
- `scipy >= 1.7.1`
- `networkx >= 2.6.3`
- `joblib >= 1.0.1`
- `matplotlib >= 3.4.3`
- `seaborn >= 0.11.2`
- `tqdm >= 4.62.0`

[//]: # ()
[//]: # (## Getting Started)

[//]: # ()
[//]: # (Here's a simple example to get you started with NS-SCMMAB:)

[//]: # ()
[//]: # (### Basic Example: Simple Non-Stationary Environment)

[//]: # ()
[//]: # (```python)

[//]: # (import numpy as np)

[//]: # (from npsem.model import CausalDiagram, StructuralCausalModel)

[//]: # (from npsem.pomis_plus import POMISplusSEQ)

[//]: # (from npsem.bandits import ThompsonSamplingBandit)

[//]: # ()
[//]: # (# Define a simple non-stationary causal structure)

[//]: # (# Time step 1: X1 -> Z1 -> Y1)

[//]: # (# Time step 2: X2 -> Z2 -> Y2, with X1 -> X2 &#40;temporal dependency&#41;)

[//]: # ()
[//]: # (# Create causal diagram)

[//]: # (cd = CausalDiagram&#40;&#41;)

[//]: # (cd.add_edges&#40;[)

[//]: # (    &#40;'X1', 'Z1'&#41;, &#40;'Z1', 'Y1'&#41;,  # Time step 1)

[//]: # (    &#40;'X2', 'Z2'&#41;, &#40;'Z2', 'Y2'&#41;,  # Time step 2)

[//]: # (    &#40;'X1', 'X2'&#41;                  # Temporal edge)

[//]: # (]&#41;)

[//]: # ()
[//]: # (# Define structural equations)

[//]: # (def f_X1&#40;u&#41;: return u['U_X1'])

[//]: # (def f_Z1&#40;x1, u&#41;: return &#40;x1 + u['U_Z1']&#41; % 2)

[//]: # (def f_Y1&#40;z1, u&#41;: return &#40;z1 + u['U_Y1']&#41; % 2)

[//]: # ()
[//]: # (def f_X2&#40;x1, u&#41;: return &#40;x1 + u['U_X2']&#41; % 2)

[//]: # (def f_Z2&#40;x2, u&#41;: return &#40;x2 + u['U_Z2']&#41; % 2)

[//]: # (def f_Y2&#40;z2, u&#41;: return &#40;z2 + u['U_Y2']&#41; % 2)

[//]: # ()
[//]: # (# Define exogenous distributions)

[//]: # (P_U = {)

[//]: # (    'U_X1': 0.5, 'U_Z1': 0.2, 'U_Y1': 0.1,)

[//]: # (    'U_X2': 0.3, 'U_Z2': 0.2, 'U_Y2': 0.1)

[//]: # (})

[//]: # ()
[//]: # (# Create SCM)

[//]: # (scm = StructuralCausalModel&#40;)

[//]: # (    graph=cd,)

[//]: # (    functions={'X1': f_X1, 'Z1': f_Z1, 'Y1': f_Y1,)

[//]: # (               'X2': f_X2, 'Z2': f_Z2, 'Y2': f_Y2},)

[//]: # (    exogenous_dist=P_U)

[//]: # (&#41;)

[//]: # ()
[//]: # (# Compute POMIS+ sequences)

[//]: # (reward_vars = ['Y1', 'Y2'])

[//]: # (pomis_plus_sequences = POMISplusSEQ&#40;cd, reward_vars, time_horizon=2&#41;)

[//]: # ()
[//]: # (print&#40;"POMIS+ Intervention Sequences:"&#41;)

[//]: # (for i, seq in enumerate&#40;pomis_plus_sequences, 1&#41;:)

[//]: # (    print&#40;f"Sequence {i}: {seq}"&#41;)

[//]: # (```)

[//]: # ()
[//]: # (### Running a Bandit Experiment)

[//]: # ()
[//]: # (```python)

[//]: # (from npsem.scm_bandits import SCMBandit)

[//]: # (from npsem.bandits import ThompsonSamplingBandit)

[//]: # ()
[//]: # (# Create a bandit problem from the SCM)

[//]: # (bandit = SCMBandit&#40;scm, reward_vars=['Y1', 'Y2']&#41;)

[//]: # ()
[//]: # (# Initialize Thompson Sampling)

[//]: # (ts = ThompsonSamplingBandit&#40;)

[//]: # (    intervention_sequences=pomis_plus_sequences,)

[//]: # (    n_trials=10000)

[//]: # (&#41;)

[//]: # ()
[//]: # (# Run the bandit algorithm)

[//]: # (cumulative_regret = ts.run&#40;bandit&#41;)

[//]: # ()
[//]: # (print&#40;f"Final cumulative regret: {cumulative_regret[-1]:.2f}"&#41;)

[//]: # (```)

[//]: # ()
[//]: # (### Visualizing Results)

[//]: # ()
[//]: # (```python)

[//]: # (import matplotlib.pyplot as plt)

[//]: # ()
[//]: # (plt.figure&#40;figsize=&#40;10, 6&#41;&#41;)

[//]: # (plt.plot&#40;cumulative_regret, label='Thompson Sampling with POMIS+'&#41;)

[//]: # (plt.xlabel&#40;'Trials'&#41;)

[//]: # (plt.ylabel&#40;'Cumulative Regret'&#41;)

[//]: # (plt.title&#40;'Performance on Non-Stationary Causal Bandit'&#41;)

[//]: # (plt.legend&#40;&#41;)

[//]: # (plt.grid&#40;True, alpha=0.3&#41;)

[//]: # (plt.show&#40;&#41;)

[//]: # (```)

## Experiments

To reproduce the experiments from the NeurIPS 2025 paper:

### Running All Experiments

```bash
# Run experiments (~2 hours on 48-core server, ~4-6 hours on typical machines)
python -m npsem.NIPS2025POMISPLUS_exp.test_nsbandit_strategies
```

This creates a `bandit_results/` directory with results for three experimental tasks:
- **Task 1**: Standard non-stationary chain structure (Fig. 3 in paper)
- **Task 2**: Non-stationary structure with collider (Fig. 4 in paper)
- **Task 3**: Complex structure with long-range dependencies (Fig. 6 in paper)

### Generating Figures

```bash
# Generate figures as in the paper
python -m npsem.NIPS2025POMISPLUS_exp.test_drawing_re
```

This produces:
- Cumulative regret plots comparing POMIS+ vs myopic POMIS
- Optimal arm selection probability plots
- Results for both Thompson Sampling and KL-UCB algorithms

### Individual Task Experiments

To run specific tasks:

```python
from npsem.NIPS2025POMISPLUS_exp import test_nsbandit_strategies

# Run only Task 1
test_nsbandit_strategies.run_task(task_id=1, n_trials=100000, n_runs=200)
```

### Customizing Experiments

You can customize the experimental parameters:

```python
from npsem.NIPS2025POMISPLUS_exp.scm_examples import create_task1_scm
from npsem.scm_bandits import run_bandit_experiment

# Create custom SCM
scm = create_task1_scm()

# Run with custom parameters
results = run_bandit_experiment(
    scm=scm,
    algorithm='thompson_sampling',  # or 'kl_ucb'
    n_trials=50000,
    n_runs=100,
    use_pomis_plus=True  # Set to False for myopic baseline
)
```

## Project Structure

```
NS-SCMMAB/
├── npsem/                             # Main package
│   ├── model.py                       # Structural Causal Model implementation
│   ├── bandits.py                     # Bandit algorithms (Thompson Sampling, KL-UCB)
│   ├── scm_bandits.py                 # SCM-specific bandit formulations
│   ├── pomis_plus.py                  # POMIS+ algorithm implementation
│   ├── where_do.py                    # POMIS identification utilities
│   ├── utils.py                       # Helper functions
│   ├── viz_util.py                    # Visualization utilities
│   └── NIPS2025POMISPLUS_exp/         # Experimental code
│       ├── test_nsbandit_strategies.py  # Main experiment runner
│       ├── test_drawing_re.py           # Figure generation
│       ├── scm_examples.py              # Task-specific SCM definitions
│       ├── construct_pomis.py           # POMIS construction utilities
│       ├── report_cum_regret_oap.py     # Result reporting
│       └── report_mean_rewards.py       # Reward analysis
├── pyproject.toml                     # Package configuration
├── LICENSE.txt                        # MIT License
└── README.md                          # This file
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{kwon2025nonstationary,
  title={Non-Stationary Structural Causal Bandits},
  author={Kwon, Yeahoon and Choe, Yesong and Park, Soungmin and Dhir, Neil and Lee, Sanghack},
  booktitle={Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025},
  url={https://openreview.net/pdf?id=F4LhOqhxkk}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

This work was supported by:
- IITP (RS-2022-II220953, RS-2025-02263754) grants funded by the Korean government
- NRF (RS-2023-00211904, RS-2023-00222663) grants funded by the Korean government
- Basic Science Research Program through the NRF funded by the Ministry of Education (RS-2025-25418030)

## Contact

For questions or issues:
- **Open an issue**: [GitHub Issues](https://github.com/yeahoon-k/NS-SCMMAB/issues)
- **Email**: sanghack@snu.ac.kr or neil.dhir@focused-energy.co

## References

### This Work
- **Paper**: [Non-Stationary Structural Causal Bandits](https://openreview.net/pdf?id=F4LhOqhxkk) (NeurIPS 2025)
- **Conference Page**: [NeurIPS 2025 Poster #119070](https://neurips.cc/virtual/2025/loc/san-diego/poster/119070)
- **Blog Post**: [Interactive Explainer](https://yeahoon-k.github.io/ns-scb-explainer/)

### Related Work
- Lee, S., & Bareinboim, E. (2018). Structural Causal Bandits: Where to Intervene? *NeurIPS 2018*.
- Lee, S., & Bareinboim, E. (2019). Structural Causal Bandits with Non-manipulable Variables. *AAAI 2019*.
- Pearl, J. (2009). *Causality: Models, Reasoning and Inference*. Cambridge University Press.
