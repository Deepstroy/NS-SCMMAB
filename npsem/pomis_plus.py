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
def POMISplusSEQ(G: CausalDiagram,
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
    sorted_POMIS_MUCT = POMISs_MUCT(G_t, Yt)

    IBplus_origin = copy.deepcopy(IBplus)
    QIB_origin = copy.deepcopy(QIB)
    for Xs, Ts in sorted_POMIS_MUCT:
        IBplus = update_IBplus(Xs, Vs, IBplus)
        QIB = update_QIB(G, Vs, Ys, IBplus, QIB, Ts)
        complete_time = min(IBplus.keys())
        if 0 < complete_time:
            POMISplusSEQ(G=G, Vs=Vs, Ys=Ys, T=complete_time - 1, IBplus=copy.deepcopy(IBplus), QIB=copy.deepcopy(QIB))
        else:
            result_combination = []
            for key in IBplus:
                if key in QIB:
                    combinations = [frozenset(set(IBplus[key]) | qib_item) for qib_item in QIB[key]]
                    result_combination.append(combinations)
            Sequences.extend(list(product(*result_combination)))

        IBplus = copy.deepcopy(IBplus_origin)
        QIB = copy.deepcopy(QIB_origin)
    return list(set(Sequences))


def update_IBplus(Xs: FrozenSet[str], Vs: List[Set], IBplus: Dict[int, List[str]]) -> Dict[int, List[str]]:
    ''' Update IBplus dictionary from Xs and Vs '''
    if not Xs:
        # Handle empty IB explicitly — assign to t = 0
        if 0 not in IBplus:
            IBplus[0] = []
        # Append nothing, but at least the key exists
        return IBplus
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
            G_Vt = G[Vs[t]].do(set(IBplus_t))
            QIB[t] = []
            for IB_t in POMISs(G_Vt, Ys[t]):
                if not (Ts & IB_t):
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

