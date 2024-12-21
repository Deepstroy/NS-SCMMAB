import numpy as np

def UCB(data: dict, actions: list[dict], reward_vars: list[str]) -> np.ndarray:
    T = len(data[0])

    actions_means = np.zeros(len(actions))
    actions_counts = np.zeros(len(actions), dtype=int)
    action_played = np.zeros(T, dtype=int)

    for t in range(T):
        ucb_values = actions_means + np.sqrt(
            2 * np.log(t + 1) / np.maximum(actions_counts, 1)
        )
        ucb_values[actions_counts == 0] = np.inf
        action_index = np.argmax(ucb_values)

        reward = data[action_index][reward_vars].iloc[t, :]
        total_reward = sum(reward)
        actions_counts[action_index] += 1
        actions_means[action_index] = (
            actions_means[action_index]
            + (total_reward - actions_means[action_index])
            / actions_counts[action_index]
        )
        action_played[t] = action_index

    return action_played


def thompson_sampling(data: dict, actions: list[dict], reward_vars: list[str]) -> np.ndarray:
    """
    TS
    Parameters:
        data : 액션별 보상 데이터
        actions : 액션 시퀀스 리스트
        reward_vars : 보상 변수들
    Returns:
        action_played (np.ndarray): 각 시간 t에서 선택된 액션 인덱스
    """
    # 총 시간 단계
    T = len(data[0])

    # Beta 분포의 파라미터 초기화 (각 arm에 대해 성공 횟수 alpha와 실패 횟수 beta)
    alpha = np.ones(len(actions))  # 성공 횟수
    beta = np.ones(len(actions))   # 실패 횟수

    action_played = np.zeros(T, dtype=int)

    for t in range(T):
        # 각 arm에 대해 Beta 분포에서 샘플링
        sampled_theta = np.random.beta(alpha, beta)

        # 샘플링된 theta 값이 가장 큰 arm 선택
        action_index = np.argmax(sampled_theta)

        # 데이터에서 보상 가져오기
        reward = data[action_index][reward_vars].iloc[t, :]
        total_reward = sum(reward)

        # Beta 분포의 파라미터 업데이트
        if total_reward > 0:  # 성공 시 alpha 증가
            alpha[action_index] += 1
        else:  # 실패 시 beta 증가
            beta[action_index] += 1

        action_played[t] = action_index

    return action_played
