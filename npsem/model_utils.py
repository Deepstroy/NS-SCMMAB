from npsem.model import CD, SCM, default_P_U
from npsem.utils import rand_bw, dict_or

from typing import Tuple, Callable, Dict, Any
from collections import defaultdict
import numpy as np


def random_scm(G: CD, P_U=None, F=None, precision=2, seed=None) -> SCM:
    if seed is not None:
        np.random.seed(seed)

    expressions = ""

    if F is None:
        expressions = "[Functions]\n"
        F = {}
        for v in sorted(G.V):
            func, expr = random_binary_func(G, v,
                                            seed)  # generate a random binary function for each variable and get expression
            F[v] = func  # assign the function to the variable
            expressions += f"\t{expr}\n"  # add the expression to the expressions string

    if P_U is None:
        mu1 = {f'U_{v}': rand_bw(0.2, 0.8, precision=precision) for v in
               sorted(G.V)}  # generate a probability for each exogenous variable

        new_Us = {f'U_{v}' for v in sorted(G.V)}
        assert G.U.isdisjoint(new_Us)  # check if the new exogenous variables are disjoint with the existing UCs.

        mu1 = dict_or(mu1, {u: rand_bw(0.2, 0.8, precision=precision) for u in
                            sorted(G.U)})  # generate a probability for each existing UCs
        P_U = default_P_U(mu1)  # generate the probability distribution for the exogenous variables
        expressions += "[P(U)]\n"
        for k, v in mu1.items():
            expressions += f"\t{k} = {v}\n"

    domains = defaultdict(lambda: (0, 1))  # type: Dict[str, Tuple[Any,...]]

    # SCM with parametrization
    return SCM(G,
               F=F,
               P_U=P_U,
               D=domains,
               more_U={f'U_{v}' for v in G.V}), expressions


def random_binary_func(G: CD, var: str, seed=None) -> Callable[[Dict[str, int]], int]:
    """ Assuming U_{var} is valid variable specific disturbance ... """
    # make U_{var} explicitly xor of the evaluated
    if seed is not None:
        np.random.seed(seed)

    # variable | UCs | exogenous variable
    pas = G.pa(var) | {u for u, xy in G.confounded_dict.items() if var in xy} | {f'U_{var}'}
    # pas = G.pa(var) | {u for u, xy in G.u2vv.items() if var in xy}

    if not pas:
        def inner2(v):
            return v[f'U_{var}'], f"{var}: U_{var}"

        return inner2

    assert pas
    pas = sorted(pas)

    # 0: and 1: or 2: xor
    operators = list(np.random.randint(0, 3, len(pas) - 1))  # 0 and, 1 or , 2 xor
    postfix = pas + operators
    np.random.shuffle(postfix)

    while True:
        cnt_numbers = 0
        for i, cell in enumerate(postfix):
            if not isinstance(cell, str):
                if cnt_numbers < 2:
                    postfix.pop(i)
                    postfix.append(cell)
                    break
                else:
                    cnt_numbers -= 1
            else:
                cnt_numbers += 1
        else:
            break

    to_not = np.random.poisson(max(len(pas) // 2, 1))
    if to_not <= len(pas):
        insert_indices = np.random.choice(len(pas), to_not, replace=False)
        insert_indices = insert_indices[insert_indices > 0]
        insert_indices = sorted(insert_indices, reverse=True)
        for at in insert_indices:
            postfix.insert(at, 3)

    # vals = {'a': 0, 'b': 0, 'c': 1, 'd': 0, 'e': 0, 'f': 1, 'g': 0}
    operators = ['and', 'or', 'xor', 'not']

    def infix_expression(postfix):
        stack = []
        ops_symbols = ['&', '|', '^', 'not']
        for item in postfix:
            if isinstance(item, str):
                stack.append(item)
            elif item == 3:  # 'not' operator
                if stack:
                    op = stack.pop()
                    # stack.append(f"{ops_symbols[item]} {op}")
                    stack.append(f"(1 - {op})")  # more readable
            else:
                if len(stack) >= 2:
                    op2 = stack.pop()
                    op1 = stack.pop()
                    stack.append(f"({op1} {ops_symbols[item]} {op2})")

        return f"{var}: " + (' '.join(stack))

    expression = infix_expression(postfix)

    def inner(v: Dict) -> int:
        array = [v[i_] if isinstance(i_, str) else operators[i_] for i_ in postfix]
        while len(array) != 1:
            for at_, cell_ in enumerate(array):
                if cell_ in operators:
                    if cell_ == 'not':
                        array[at_ - 1] = 1 - array[at_ - 1]
                        array.pop(at_)
                        break
                    elif cell_ == 'and':
                        array[at_ - 2] = array[at_ - 2] & array[at_ - 1]
                        array.pop(at_)
                        array.pop(at_ - 1)
                        break
                    elif cell_ == 'xor':
                        array[at_ - 2] = array[at_ - 2] ^ array[at_ - 1]
                        array.pop(at_)
                        array.pop(at_ - 1)
                        break
                    elif cell_ == 'or':
                        array[at_ - 2] = array[at_ - 2] | array[at_ - 1]
                        array.pop(at_)
                        array.pop(at_ - 1)
                        break
        return array[0]

    return inner, expression