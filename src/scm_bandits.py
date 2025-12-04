from itertools import product
from collections import defaultdict
from typing import Dict, Tuple, Union, Any

from npsem.model import StructuralCausalModel
from npsem.utils import combinations
from npsem.where_do import POMISs
from npsem.pomis_plus import POMISplusSEQ


def group_by_time_index(G_V):
    grouped = defaultdict(set)
    for var in G_V:
        for i in range(len(var)):
            if var[i:].isdigit():
                idx = int(var[i:])
                grouped[idx].add(var)
                break
    return [grouped[i] for i in sorted(grouped.keys())]

def SCM_to_bandit_machine(M: StructuralCausalModel, Ys: set()) -> Tuple[Tuple, Dict[Union[int, Any], Dict]]:
    """
    Returns all intervention combinations for standard arms (e.g., {'0': {}, '1': {'S': 0, 'T': 1}, ...})
    and their true expected rewards (mu)
    """
    G = M.G
    mu_arm = list()
    arm_setting = dict()
    arm_id = 0
    all_subsets = list(combinations(sorted(G.V - Ys)))
    for _, subset in enumerate(all_subsets):

        # Cartesian product of variable domains, e.g., X0: D(0,1), Z0: D(0,1) → (0,0), (0,1), (1,0), (1,1)
        for values in product(*[M.D[variable] for variable in subset]):

            arm_setting[arm_id] = dict(zip(subset, values))

            Ys = tuple(Ys)
            result = M.query(Ys, intervention=arm_setting[arm_id])

            expectation = 0
            for y_values, p_ys in result.items():
                expectation += sum(y_values) * p_ys
            mu_arm.append(expectation)
            arm_id += 1

    return tuple(mu_arm), arm_setting

def ns_arm_types():
    return ['POMIS', 'POMIS+']

def ns_arms_of(arm_type: str, arm_setting, G, Ys) -> Tuple[int, ...]:
    if arm_type == 'POMIS+':
        return pomis_plus_arms_of(arm_setting, G, Ys) # tuple of arm number (e.g., 0, 24, 53, ..)
    elif arm_type == 'POMIS':
        return pomis_arms_of(arm_setting, G, Ys)
    elif arm_type == 'Brute-force':
        return tuple(range(len(arm_setting)))
    raise AssertionError(f'unknown: {arm_type}')

def pomis_plus_arms_of(arm_setting, G, Ys):
    Vs = group_by_time_index(G.V)

    def key_by_time(y):
        return int(''.join(filter(str.isdigit, y)))

    Ys = sorted(Ys, key=key_by_time)
    re = POMISplusSEQ(G=G, Vs=Vs, Ys=list(Ys), T=len(Ys) - 1)
    pomis_pluss = {frozenset().union(*tup) for tup in re}

    return tuple(arm_x for arm_x in range(len(arm_setting)) if set(arm_setting[arm_x]) in pomis_pluss)


def pomis_arms_of(arm_setting, G, Ys):
    Vs = group_by_time_index(G.V)
    pomiss_ts = dict()
    for t, V_t in enumerate(Vs):
        pomiss_ts[t] = POMISs(G[V_t],'Y'+str(t))

    # calculate each pomis for each time (Myopic)
    pomiss_myoptic = {frozenset().union(*combo) for combo in product(*[pomiss_ts[t] for t in sorted(pomiss_ts)])}

    return tuple(arm_x for arm_x in range(len(arm_setting)) if set(arm_setting[arm_x]) in pomiss_myoptic)