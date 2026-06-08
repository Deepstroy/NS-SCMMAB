import multiprocessing
import numpy as np
import os
from pathlib import Path

from npsem.NIPS2025POMISPLUS_exp.scm_examples import X0toY2, WttoYtprime, W0toY2
from npsem.bandits import play_bandits
from npsem.model import StructuralCausalModel
from npsem.scm_bandits import SCM_to_bandit_machine, ns_arm_types, ns_arms_of
from npsem.utils import subseq, mkdirs


def main_experiment(M: StructuralCausalModel, Ys: set(), num_trial=200, horizon=10000, n_jobs=1):
    results = dict()

    # mu: expected rewards of all arms
    # arm_setting: combinations of realizations for each intervention set (e.g., {'0': {}, '1':{'S' : 0, 'T' :1},...})
    mu, arm_setting = SCM_to_bandit_machine(M, Ys=Ys)

    for arm_strategy in ns_arm_types():

        # selected arm index from the strategy (POMIS/POMIS+) in the arm_setting (e.g., 11, 12, 13, 14, 27...)
        arm_selected = ns_arms_of(arm_strategy, arm_setting, M.G, Ys)

        # mapping function (e.g., arm_corrector(1) => arm_setting[X] = 12)
        arm_corrector = np.vectorize(lambda x: arm_selected[x])

        for bandit_algo in ['TS', 'UCB']:
            # subseq(mu, arm_selected) : extract the expected reward corresponding to the arm_selected from the mu
            arm_played, rewards = play_bandits(horizon, subseq(mu, arm_selected), bandit_algo, num_trial, n_jobs)
            results[(arm_strategy, bandit_algo)] = arm_corrector(arm_played), rewards

    return results, mu

def compute_arm_frequencies(arm_played, num_arms, horizon=None):
    if horizon is not None:
        arm_played = arm_played[:, :horizon]

    counts = np.zeros((len(arm_played), num_arms))
    for i in range(num_arms):
        counts[:, i] = np.mean((arm_played == i).astype(int), axis=1)
    return counts

def compute_optimality(arm_played, mu):
    mu_star = np.max(mu)
    return np.vectorize(lambda x: int(mu[x] == mu_star))(arm_played)

def compute_cumulative_regret(arm_played: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu_star = np.max(mu)
    regret_matrix = mu_star - mu[arm_played]  # (num_trials, horizon)
    cumulative_regret = np.cumsum(regret_matrix, axis=1)
    return cumulative_regret

def load_result(directory):
    results = dict()
    print(directory)
    for arm_strategy in ns_arm_types():
        for bandit_algo in ['TS', 'UCB']:
            loaded = np.load(directory + f'/{arm_strategy}---{bandit_algo}.npz', allow_pickle=True)
            arms = loaded['a']
            rewards = loaded['b']
            results[(arm_strategy, bandit_algo)] = (arms, rewards)

    p_u = np.load(directory + '/p_u.npz', allow_pickle=True)['a'][()]
    mu = np.array(np.load(directory + '/mu.npz', allow_pickle=True)['a'])

    return p_u, mu, results

def save_result(directory, p_u, mu, results):
    mkdirs(directory)
    for arm_strategy, bandit_algo in results:
        arms, rewards = results[(arm_strategy, bandit_algo)]
        np.savez_compressed(directory + f'/{arm_strategy}---{bandit_algo}', a=arms, b=rewards)
    np.savez_compressed(directory + f'/p_u', a=p_u)
    np.savez_compressed(directory + f'/mu', a=mu)

def finished(directory, flag=None, message=''):
    mkdirs(directory)
    filename = directory + '/finished.txt'
    if flag is not None:
        if flag:
            Path(filename).touch(exist_ok=True)
            with open(filename, 'w') as f:
                print(str(message), file=f)
            return True
        else:
            os.remove(filename)
            return False
    else:
        return os.path.exists(filename)

def main():
    model_w0toy2, p_u_w0toy2 = W0toY2(True, seed=0)
    model_wttoytprime, p_u_wttoytprime = WttoYtprime(True, seed=0)
    model_x0toy2, p_u_x0toy2 = X0toY2(True, seed=0)
    num_simulation_repeats = 200

    for dirname, (model, p_u), horizon, Ys in [
                                            ('WttoYtprime', (model_wttoytprime, p_u_wttoytprime) , 100000, {'Y0', 'Y1'}),
                                            ('X0toY2', (model_x0toy2, p_u_x0toy2), 100000, {'Y0', 'Y1', 'Y2'}),
                                            ('W0toY2', (model_w0toy2, p_u_w0toy2) , 100000, {'Y0', 'Y1', 'Y2'}),
    ]:
        BASE_DIR = Path(__file__).resolve().parents[2]
        directory = str(BASE_DIR / f'bandit_results/{dirname}_0')
        print("BASE_DIR:", BASE_DIR)
        print("directory:", directory)

        results, mu = main_experiment(model, Ys, num_simulation_repeats, horizon, n_jobs=multiprocessing.cpu_count())
        save_result(directory, p_u, mu, results)
        finished(directory, flag=True)

if __name__ == '__main__':
    main()
