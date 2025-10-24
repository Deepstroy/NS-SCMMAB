import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

from npsem.NIPS2025POMISPLUS_exp.test_nsbandit_strategies import load_result, compute_cumulative_regret, compute_optimality
from npsem.utils import with_default
from npsem.viz_util import sparse_index
from matplotlib.ticker import FuncFormatter

def k_format(x, pos):
    if x >= 1000:
        return f"{int(x/1000)}K"
    else:
        return f"{int(x)}"

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{helvet}\usepackage{sansmath}\sansmath"

c__ = sns.color_palette('Set1', 4)
COLORS = [c__[0], c__[0], c__[1], c__[1], c__[2], c__[2], c__[3], c__[3]]


def naked_MAB_regret_plot(axes: Axes, xs_dict, cut_time, band_alpha=0.1, legend=False, hide_ylabel=False, adjust_ymax=1, hide_yticklabels=False, **_kwargs):
    for i, (name, value_matrix) in list(enumerate(xs_dict.items())):
        mean_x = np.mean(value_matrix, axis=0)
        sd_x = np.std(value_matrix, axis=0)
        lower, upper = mean_x - sd_x, mean_x + sd_x

        time_points = sparse_index(with_default(cut_time, len(mean_x)), 200)
        axes.plot(time_points, mean_x[time_points], lw=1, label=name.split(' ')[0] if '(TS)' in name else None, color=COLORS[i], linestyle='-' if '(TS)' in name else '--')
        axes.fill_between(time_points, lower[time_points], upper[time_points], color=COLORS[i], alpha=band_alpha, lw=0)

    axes.xaxis.set_major_formatter(FuncFormatter(k_format))
    axes.yaxis.set_major_formatter(FuncFormatter(k_format))

    if legend:
        axes.legend(loc=2, frameon=False)
    if not hide_ylabel:
        axes.set_ylabel('Cumulative\nRegret')
        axes.get_yaxis().set_label_coords(-0.17, 0.5) #-0.15
    if adjust_ymax != 1:
        ymin, ymax = axes.get_ylim()
        axes.set_ylim(ymin, ymax * adjust_ymax)
    if hide_yticklabels:
        axes.set_yticklabels([])


def naked_MAB_optimal_probability_plot(axes: Axes, arm_freqs, cut_time, legend=False, hide_ylabel=False, hide_yticklabels=False, **_kwargs):
    for i, (name, arm_freq) in list(enumerate(arm_freqs.items())):
        time_points = sparse_index(with_default(cut_time, len(arm_freq)), 200)
        axes.plot(time_points, arm_freq[time_points], lw=1, label=name.split(' ')[0] if '(TS)' in name else None, color=COLORS[i], linestyle='-' if '(TS)' in name else '--')

    axes.xaxis.set_major_formatter(FuncFormatter(k_format))

    if legend:
        axes.legend(loc=4, frameon=False)
    axes.set_xlabel('Trials')
    if not hide_ylabel:
        axes.set_ylabel('Probability')
        axes.get_yaxis().set_label_coords(-0.2, 0.5)
    axes.set_yticks([0, 0.5, 1.0])
    if hide_yticklabels:
        axes.set_yticklabels([])

    axes.set_ylim(-0.05, 1.02)


def data_prep(directory):
    _, mu, results = load_result(directory)
    mu_star = np.max(mu)

    regret_results = dict()
    arm_optimality_results = dict()

    # prepare data
    for (arm_strategy, bandit_algo), (arm_played, rewards) in results.items():
        legend_label = arm_strategy + ' (' + bandit_algo + ')'

        # cumulative_regret = compute_cumulative_regret(rewards, mu_star)
        cumulative_regret = compute_cumulative_regret(arm_played, mu)
        arm_optimality = compute_optimality(arm_played, mu)

        regret_results[legend_label] = cumulative_regret
        arm_optimality_results[legend_label] = np.mean(arm_optimality, axis=0)

    return regret_results, arm_optimality_results

def aggregate_plot():
    """ Prepare data """
    info__ = {
        "Task 1": (absolute_path + 'bandit_results/X0toY2_0', 100000),
        "Task 2": (absolute_path + 'bandit_results/WttoYtprime_0', 100000),
        "Task 3": (absolute_path + 'bandit_results/W0toY2_0', 100000)
    }

    # info dict
    info = {k: dict(zip(['directory', 'cut_time'], v)) for k, v in info__.items()}
    results = {task_name: dict(zip(['CR', 'OAP'], data_prep(task_info['directory']))) for task_name, task_info in info.items()}
    plot_funcs = {'CR': naked_MAB_regret_plot, 'OAP': naked_MAB_optimal_probability_plot}

    """ Start drawing """
    fig, ax = plt.subplots(2, 3, sharex='col', figsize=(8, 3.25))

    task_names = list(info.keys())
    for row_id, plot_type in enumerate(['CR', 'OAP']):
        for col_id in range(3):
            current_axes = ax[row_id, col_id]
            if col_id < len(task_names):
                task_name = task_names[col_id]
                plot_funcs[plot_type](current_axes,
                                      results[task_name][plot_type],
                                      info[task_name]['cut_time'],
                                      legend=(row_id == 1 and col_id == 0),
                                      hide_ylabel=(col_id != 0),
                                      hide_yticklabels=False,
                                      adjust_ymax=1)
                if row_id == 1:
                    current_axes.text(0.5, -0.45, f"({chr(97 + col_id)}) {task_name}",
                                      transform=current_axes.transAxes,
                                      ha='center', va='top', fontsize=10, family='serif')
            else:
                current_axes.axis('off')

    sns.despine(fig)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.175)
    fig.savefig('aggregate_single.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()


def print_final_bands(directory, round_t=None, ci_scale=1.96):
    _, mu, results = load_result(directory)

    print(f"\n=== Final ±95% CI @ {directory} ===")
    for (arm_strategy, bandit_algo), (arm_played, rewards) in results.items():
        name = f"{arm_strategy} ({bandit_algo})"

        # --- Cum.Regret ---
        cr_matrix = compute_cumulative_regret(arm_played, mu)
        T = cr_matrix.shape[1]
        t_final = (round_t if round_t is not None else T)
        t_final = min(t_final, T)
        cr_last = cr_matrix[:, t_final-1]

        n_trials = len(cr_last)
        cr_mean = np.mean(cr_last)
        cr_sd   = np.std(cr_last, ddof=1)
        cr_se   = cr_sd / np.sqrt(n_trials)
        cr_pm   = ci_scale * cr_se

        # --- OAP ---
        opt_matrix = compute_optimality(arm_played, mu).astype(float)
        oap_last = opt_matrix[:, t_final-1]
        n_trials_oap = len(oap_last)
        oap_mean = np.mean(oap_last)
        oap_sd   = np.std(oap_last, ddof=1)
        oap_se   = oap_sd / np.sqrt(n_trials_oap)
        oap_pm   = ci_scale * oap_se

        # --- print ± form ---
        print(f"{name:16s} | CR: {cr_mean:8.3f} ± {cr_pm:7.3f}"
              f"   | OAP: {100*oap_mean:6.2f}% ± {100*oap_pm:5.2f}%")



if __name__ == '__main__':
    absolute_path = '../../'

    aggregate_plot()

    print_final_bands(absolute_path + 'bandit_results/X0toY2_0', round_t=100000, ci_scale=1.96)
    print_final_bands(absolute_path + 'bandit_results/WttoYtprime_0', round_t=100000, ci_scale=1.96)
    print_final_bands(absolute_path + 'bandit_results/W0toY2_0', round_t=100000, ci_scale=1.96)
