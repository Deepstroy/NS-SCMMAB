import tqdm
import pandas as pd
from itertools import product
from joblib import Parallel, delayed

from npsem.model import CD
from npsem.model_utils import random_scm
from npsem.viz_util import plot_regrets
from npsem.model import SCM
from npsem.ns_bandits_solver import *


def create_multiple_timestep_graph(G: CD, transition_edges: list[tuple], t: int):
    variables = G.V
    UCs = G.confounded_to_3tuples()
    edges = G.edges

    new_variables = []
    new_UCs = []
    new_edges = []
    for i in range(t):
        for v in variables:
            new_variables.append(f"{v}{i}")
        for var_a, var_b, UC in UCs:
            new_UCs.append((f"{var_a}{i}", f"{var_b}{i}", f"{UC}_{i}"))
        for edge in edges:
            new_edges.append((f"{edge[0]}{i}", f"{edge[1]}{i}"))

    new_G = CD(new_variables, new_edges, new_UCs)
    new_G += transition_edges

    return new_G


def create_all_actions(arms: list[tuple]) -> list[dict]:
    """
    create all binary actions given the set of arms
    """
    arms.sort()
    all_actions = []
    for arm in arms:
        for comb in product((0, 1), repeat=len(arm)):
            action = dict(zip(arm, comb))
            all_actions.append(action)
    return all_actions


def calculate_expected_rewards(
    M: SCM, actions: list[dict], reward_vars: list[str]
) -> list[float]:
    """
    calculate the expected reward for each action
    """
    reward_vars = tuple(reward_vars)
    expected_rewards = np.zeros(len(actions))
    for idx, action in enumerate(actions):
        result = M.query(outcome=reward_vars, intervention=action)
        val = 0
        for k, v in result.items():
            val += sum(k) * v
        expected_rewards[idx] = val
    return expected_rewards


def sampling_from_SCM(M: SCM, size: int, intervention: dict) -> dict:
    """
    sample from SCM given the intervention
    """
    U = list(sorted(M.G.U | M.more_U))  # all exogenous variables
    D = M.D  # domain of each variable
    P_U = M.P_U  # distribution of exogenous variables
    V_ordered = M.G.causal_order()

    # TODO save time here?
    u_vals = {tuple(u): P_U(dict(zip(U, u))) for u in product(*[D[U_i] for U_i in U])}
    u_configs = tuple(u_vals.keys())  # order preserved! 3.7!
    unit_counts = np.random.multinomial(size, list(u_vals.values()))
    data = np.zeros((size, len(V_ordered)))
    offset = 0
    for ith_unit, how_many in enumerate(unit_counts):
        if how_many == 0:
            continue
        u = u_configs[ith_unit]
        assigned = dict(zip(U, u))
        for V_i in V_ordered:
            if V_i in intervention:
                assigned[V_i] = intervention[V_i]
            else:
                assigned[V_i] = M.F[V_i](assigned)

        generated = np.array([assigned[V_i] for V_i in V_ordered])
        data[offset : (offset + how_many), :] = generated
        offset += how_many
    np.random.shuffle(data)
    sampled_data = pd.DataFrame(data, columns=V_ordered)
    return sampled_data


def data_generator(M: SCM, actions: list[dict], T: int) -> dict[int, pd.DataFrame]:
    """
    generate data from SCM given the actions
    """
    all_data = dict()
    for idx, action in enumerate(actions):
        sampled_data = sampling_from_SCM(M, T, action)
        all_data[idx] = sampled_data

    return all_data


def calculate_expected_regret(expected_rewards, arm_played):
    """
    calculate the expected regret
    """
    T = len(arm_played)
    reward = 0
    for t in range(T):
        reward += expected_rewards[arm_played[t]]

    best_reward = np.max(expected_rewards)
    cumulative_regret = T * best_reward - reward

    return cumulative_regret



def single_simulation_ucb1(Tvals, M, actions, reward_vars):
    # Tvals : the number of data generated
    ucb_expected_regrets = np.zeros(len(Tvals))

    for i, T in tqdm.tqdm(enumerate(Tvals), total=len(Tvals), desc="Single Simulation"):
        data = data_generator(M, actions, T)
        expected_rewards = calculate_expected_rewards(M, actions, reward_vars)

        # ucb1 regert
        ucb_arms = UCB(data, actions, reward_vars)
        ucb_expected_regrets[i] = calculate_expected_regret(expected_rewards, ucb_arms)

    return ucb_expected_regrets


def single_simulation_klucb(Tvals, M, actions, reward_vars):
    # Tvals : the number of data generated
    kl_ucb_expected_regrets = np.zeros(len(Tvals))

    for i, T in tqdm.tqdm(enumerate(Tvals), total=len(Tvals), desc="Single Simulation"):
        data = data_generator(M, actions, T)
        expected_rewards = calculate_expected_rewards(M, actions, reward_vars)

        # kl-ucb regert
        ucb_arms = kl_ucb(data, actions, reward_vars)
        kl_ucb_expected_regrets[i] = calculate_expected_regret(expected_rewards, ucb_arms)

    return kl_ucb_expected_regrets


def single_simulation_TS(Tvals, M, actions, reward_vars):
    # Tvals : the number of data generated
    TS_expected_regrets = np.zeros(len(Tvals))

    for i, T in tqdm.tqdm(enumerate(Tvals), total=len(Tvals), desc="Single Simulation"):
        data = data_generator(M, actions, T)
        expected_rewards = calculate_expected_rewards(M, actions, reward_vars)

        # kl-ucb regert
        ucb_arms = thompson_sampling(data, actions, reward_vars)
        TS_expected_regrets[i] = calculate_expected_regret(expected_rewards, ucb_arms)

    return TS_expected_regrets


def run_simulation(
    Tvals: np.ndarray, M: SCM, actions: list[dict], reward_vars: list[str], avg: int
):
    # Run multiple simulations to get the average regret
    results_ucb1 = Parallel(n_jobs=-1)(
        delayed(single_simulation_ucb1)(Tvals, M, actions, reward_vars)
        for _ in tqdm.tqdm(range(avg), desc="Running Simulations-ucb1")
    )
    ucb1_expected_regrets = np.sum(results_ucb1, axis=0)
    avg_ucb1_expected_regrets = ucb1_expected_regrets / avg

    results_TS = Parallel(n_jobs=-1)(
        delayed(single_simulation_TS)(Tvals, M, actions, reward_vars)
        for _ in tqdm.tqdm(range(avg), desc="Running Simulations-TS")
    )
    TS_expected_regrets = np.sum(results_TS, axis=0)
    avg_TS_expected_regrets = TS_expected_regrets / avg

    return avg_ucb1_expected_regrets, avg_TS_expected_regrets


if __name__ == "__main__":
    # Define Graph and SCM
    G = CD(["X", "Z", "Y"], [("X", "Z"), ("Z", "Y")], [("X", "Y", "U_XY")])
    transition_edges = [("X0", "X1"), ("X1", "X2")]
    reward_vars = ["Y0", "Y1", "Y2"]

    # Multi-Graph G corresponding with SCM M
    multiple_G = create_multiple_timestep_graph(G, transition_edges, 3)
    multiple_M, _ = random_scm(multiple_G, seed=1)

    # POMIS+: supposed to be set of POMIS+
    pomis_plus_arms = [('Z0', 'Z1', 'Z2'),
            ('Z0', 'Z1', 'X2', 'Z2'),
            ('Z0', 'X1', 'Z1'),
            ('Z1', 'Z2'),
            ('Z1', 'X2', 'Z2'),
            ('X1', 'Z1'),
            ('X0', 'Z0', 'Z2'),
            ('X0', 'Z0', 'X2', 'Z2')]
    actions = create_all_actions(pomis_plus_arms)

    Tvals = np.arange(1000, 150000, 20000)
    avg = 48 # <-best

    results = run_simulation(Tvals, multiple_M, actions, reward_vars, avg)
    plot_regrets(Tvals, results)

