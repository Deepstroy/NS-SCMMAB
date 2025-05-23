import numpy as np


def compute_cr_oap_at_round(mu, arm_played, round_t):
    mu_star = np.max(mu)

    # cumulative regret
    regret_matrix = mu_star - mu[arm_played[:, :round_t]]
    avg_cum_regret = np.mean(np.sum(regret_matrix, axis=1))

    # optimality
    optimal_matrix = (mu[arm_played[:, :round_t]] == mu_star).astype(float)
    avg_oap = 100 * np.mean(optimal_matrix)

    return avg_cum_regret, avg_oap

if __name__ == '__main__':
    absolute_path = '../../'  # Adjust if needed
    exp_dirs = {
        'X0toY2': absolute_path + 'bandit_results/X0toY2_0',
        'WttoYtprime': absolute_path + 'bandit_results/WttoYtprime_0',
        'W0toY2': absolute_path + 'bandit_results/W0toY2_0',
    }

    round_t = 100000

    for exp_name, dir_path in exp_dirs.items():
        print(f"\n### Results for {exp_name} ###")

        mu = np.load(f'{dir_path}/mu.npz', allow_pickle=True)['a']

        for strategy in ['POMIS+', 'POMIS']:
            for algo in ['TS', 'UCB']:
                file_path = f'{dir_path}/{strategy}---{algo}.npz'
                data = np.load(file_path)
                arm_played = data['a']
                cr, oap = compute_cr_oap_at_round(mu, arm_played, round_t)
                print(f"{strategy} - {algo}: CR@{round_t} = {cr:.2f}, OAP@{round_t} = {oap:.2f}%")