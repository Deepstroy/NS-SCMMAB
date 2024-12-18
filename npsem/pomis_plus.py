import sys
import os
import copy
import re
from typing import FrozenSet, List, Set, Tuple, Dict
from npsem.where_do import POMISs, POMISs_MUCT
from npsem.model import CD, CausalDiagram
from itertools import product
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

Sequences: List[Tuple[frozenset]] = list()


# a = POMISplus(G[V0|V1|V2|V3], Vs=Vs, Ys=['Y0', 'Y1', 'Y2', 'Y3'], T=3)
def POMISplus(G: CausalDiagram,
              Vs: List[Set],
              Ys: List[str],
              T: int,
              IBplus: Dict[int, List[str]] = None,
              QIB: Dict[int, List[frozenset]] = None) -> List[tuple[FrozenSet[str]]]:
    ''' all POMISplus sequences for G with respect to a set of time series Ys '''

    if IBplus is None: IBplus = dict()
    if QIB is None: QIB = dict()

    Yt = Ys[T]
    G_t = G[G.An(Yt)]
    all_POMISs_MUCT = POMISs_MUCT(G_t, Yt)
    sorted_POMIS_MUCT = sorted(all_POMISs_MUCT,
                               key=lambda s: min(find_timestep(item, Vs) for item in s[0]))

    IBplus_origin = copy.deepcopy(IBplus)
    QIB_origin = copy.deepcopy(QIB)
    for Xs, Ts in sorted_POMIS_MUCT:
        # update IBplus and QIB
        IBplus = update_IBplus(Xs, Vs, IBplus)
        QIB = update_QIB(G, Vs, Ys, IBplus, QIB, Ts)

        complete_time = min(IBplus.keys())
        if 0 < complete_time:
            POMISplus(G=G, Vs=Vs, Ys=Ys, T=complete_time-1, IBplus=copy.deepcopy(IBplus), QIB=copy.deepcopy(QIB))
        else:
            # combine IBplus with QIB
            result_combination = []
            for key in IBplus:
                if key in QIB:
                    # Possible all combinations
                    combinations = [frozenset(set(IBplus[key]) | qib_item) for qib_item in QIB[key]]
                    result_combination.append(combinations)
            Sequences.extend(list(product(*result_combination)))

        # 한번의 MUCT/IB가 끝났고, 다음 step에서는 더 작은 MUCT/IB가 선택되어야함. 해당 recursive의 초기 IBplus/QIB를 복구해줌
        IBplus = copy.deepcopy(IBplus_origin)
        QIB = copy.deepcopy(QIB_origin)
    return list(set(Sequences))


def update_IBplus(Xs: FrozenSet[str], Vs: List[Set], IBplus: Dict[int, List[str]]) -> Dict[int, List[str]]:
    ''' Update IBplus dictionary from Xs and Vs '''
    for X in Xs:
        t = find_timestep(X, Vs)
        if t not in IBplus:
            IBplus[t] = []
        IBplus[t].append(X)
    return IBplus


def update_QIB(G: CausalDiagram, Vs: List[Set], Ys: List[str], IBplus: Dict[int, List[str]],
               QIB: Dict[int, List[FrozenSet[str]]], Ts: FrozenSet[str]) -> Dict[int, List[FrozenSet[str]]]:
    ''' Update QIB dictionary based on IBplus '''
    for t, IBplus_t in IBplus.items():
        if t not in QIB:
            G_Vt = G[Vs[t]].do(set(IBplus_t))  # Remove incoming edges
            QIB[t] = []
            for IB_t in POMISs(G_Vt, Ys[t]):  # Generate IB sets
                if not (Ts & IB_t):  # Ensure no overlap between Ts and IB_t
                    QIB[t].append(IB_t)
    return QIB


def find_timestep(X, Vs):
    t = next((ind for ind, time_slice_nodes in enumerate(Vs) if X in time_slice_nodes), None)
    assert t is not None, f"{X} is not found in any time slice nodes"
    return t


def sort_all(A):
    ''' sort each element index 0 to 1 '''
    def extract_index(frozenset_elem):
        for elem in frozenset_elem:
            match = re.search(r'\d+', elem)
            if match:
                return int(match.group())
        return float('inf')

    def sort_tuple_frozensets(tup):
        return tuple(sorted(tup, key=extract_index))

    return [sort_tuple_frozensets(tup) for tup in A]


if __name__ == "__main__":
    V0 = {'Y0', 'Z0', 'X0', 'W0'}
    V1 = {'Y1', 'Z1', 'X1', 'W1'}
    V2 = {'Y2', 'Z2', 'X2', 'W2'}
    V3 = {'Y3', 'Z3', 'X3', 'W3'}
    Vs = [V0, V1, V2, V3]

    G = CD(V0 | V1 | V2 | V3,
           [('W0', 'X0'), ('X0', 'Z0'), ('Z0', 'Y0'),
            ('W1', 'X1'), ('X1', 'Z1'), ('Z1', 'Y1'),
            ('W2', 'X2'), ('X2', 'Z2'), ('Z2', 'Y2'),
            ('W3', 'X3'), ('X3', 'Z3'), ('Z3', 'Y3'),
            ('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3')],
           [('X0', 'Y0', 'U_X0Y0'), ('X1', 'Y1', 'U_X1Y1'),
            ('X2', 'Y2', 'U_X2Y2'), ('X3', 'Y3', 'U_X3Y3'),
            ('X1', 'X2', 'U_X1X2'), ('X1', 'Z2', 'U_X1Z2'),
            ('X2', 'X3', 'U_X2X3'), ('X2', 'Z3', 'U_X2Z3')])

    # [POMISplus test]
    # 4-steps
    # print(f"POMISs(G[V0|V1|V2|V3]): {POMISplus(G[V0|V1|V2|V3], Vs=Vs, Ys=['Y0', 'Y1', 'Y2', 'Y3'], T=3)}")
    # a = POMISplus(G[V0|V1|V2|V3], Vs=Vs, Ys=['Y0', 'Y1', 'Y2', 'Y3'], T=3)

    # 3-steps
    # print(f"POMISs(G[V0|V1|V2]: {POMISplus(G[V0|V1|V2], Vs=Vs[:3], Ys=[‘Y0’, ‘Y1’, ‘Y2’], T=2)}")
    # for ele in POMISplus(G[V0|V1|V2], Vs=Vs[:3], Ys=['Y0', 'Y1', 'Y2'], T=2):
    #     print(ele)
    # a = POMISplus(G[V0|V1|V2], Vs=Vs[:3], Ys=['Y0', 'Y1', 'Y2'], T=2)

    # 2-steps
    # print(f”POMISs(G[V0|V1]: {POMISplus(G[V0 | V1], Vs=Vs[:2], Ys=[‘Y0’, ‘Y1’], T=1)}“)
    # a = POMISplus(G[V0 | V1], Vs=Vs[:2], Ys=['Y0', 'Y1'], T=1)

    # [EMERGENCY] error return graph
    V0 = {'Y0', 'Z0', 'X0'}
    V1 = {'Y1', 'Z1', 'X1'}
    V2 = {'Y2', 'Z2', 'X2'}
    Vs = [V0, V1, V2]

    G = CD(V0 | V1 | V2,
           [('X0', 'Z0'), ('Z0', 'Y0'),
            ('X1', 'Z1'), ('Z1', 'Y1'),
            ('X2', 'Z2'), ('Z2', 'Y2'),
            ('X0', 'X1'), ('X1', 'X2')],
           [('X0', 'Y0', 'U_X0Y0'),
            ('X1', 'Y1', 'U_X1Y1'),
            ('X2', 'Y2', 'U_X2Y2')])

    a = POMISplus(G[V0|V1|V2], Vs=Vs, Ys=['Y0', 'Y1', 'Y2'], T=2)


    # ascending sort
    cnt = 0
    for _, ele in enumerate(sort_all(a)):
        print(ele)
        cnt += 1
    print(cnt)
