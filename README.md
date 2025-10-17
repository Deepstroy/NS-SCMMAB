# Non-Stationary Structural Causal Bandits
Yeahoon Kwon, Yesong Choe, Soungmin Park, Neil Dhir, and Sanghack Lee
**Non-Stationary Structural Causal Bandits**
*Thirty-Ninth Conference on Neural Information Processing Systems (**NeurIPS 2025**)*

We provide codebase to allow readers to reproduce our experiments. This code also contains various utilities related to causal diagram, structural causal model, and multi-armed bandit problem.
The code is tested with the following configuration: `python=3.9`, `numpy=1.21.2`, `scipy=1.7.1`, `joblib=1.0.1`, `matplotlib=3.4.3`, `seaborn=0.11.2`, and `networkx=2.6.3`, on
Linux and MacOS machines.


Please run the following command to perform experiments:
> `python3 -m npsem.NIPS2025POMISPLUS_exp.test_nsbandit_strategies`

This takes about less than 2 hrs in a server with 48 cores.
This will create `bandit_results` directory and there will be three directories corresponding to each task in the paper.
Then, run the following to create a figure as in the paper:
> `python3 -m npsem.NIPS2025POMISPLUS_exp.test_drawing_re`
