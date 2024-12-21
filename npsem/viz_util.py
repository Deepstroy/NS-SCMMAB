import numpy as np
import matplotlib.pyplot as plt

""" Drawing some plots for MAB or etc. """


def sparse_index(length, base_size=100):
    if length <= 2 * base_size:
        return np.arange(length)
    step = length // base_size  # >= 2
    if length % step == 0:
        temp = np.arange(1 + (length // step)) * step  # include length
        temp[-1] = length - 1
        return temp
    else:
        if (length // step) * step == length - 1:
            return np.arange(1 + (length // step)) * step
        else:
            temp = np.arange(2 + (length // step)) * step
            assert temp[-2] < length - 1
            temp[-1] = length - 1
            return temp


#
# def plot_regrets(Tvals, results):
#     plt.figure()
#     plt.plot(
#         Tvals,
#         results,
#         label=solver,
#         color="blue",
#         marker="s",
#         markersize=3,
#         linestyle="-",
#         linewidth=1,
#     )
#     # plt.title("Cumulative Regrets")
#     plt.xlabel("Trial")
#     plt.ylabel("Cum. Regrets")
#     plt.legend(loc="upper left")
#     plt.grid(True)
#     plt.savefig(f"figures/{solver}.png")
#     plt.show()

def plot_regrets(Tvals, results):
    avg_ucb1_expected_regrets, avg_TS_expected_regrets = results

    plt.figure(figsize=(8, 6))

    # UCB1: 녹색 실선
    plt.plot(Tvals, avg_ucb1_expected_regrets, label="UCB1", color="#2ca02c",
             linestyle="-", linewidth=2)

    # TS: 파란색 실선
    plt.plot(Tvals, avg_TS_expected_regrets, label="Thompson Sampling",
             color="#1f77b4", linestyle="-", linewidth=2)

    # Confidence Intervals (optional 예시로 넣음, 조정 필요)
    plt.fill_between(Tvals, avg_ucb1_expected_regrets - 2, avg_ucb1_expected_regrets + 2,
                     color="#2ca02c", alpha=0.2)
    plt.fill_between(Tvals, avg_TS_expected_regrets - 2, avg_TS_expected_regrets + 2,
                     color="#1f77b4", alpha=0.2)

    # 그래프 설정
    plt.title("Cumulative Regrets")
    plt.xlabel("Trial")
    plt.ylabel("Cum Regret")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    # 저장 및 출력
    plt.tight_layout()
    plt.savefig("/home/yeahoon/NS-SCMMAB/npsem/figures/cumulative_regrets.png", dpi=300)
    plt.show()
