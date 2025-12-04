from collections import defaultdict

from npsem.model import CausalDiagram, StructuralCausalModel, default_P_U, CD
from npsem.utils import rand_bw, seeded
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

# for graph of Fig. 1
def X0toY2(devised=True, seed=None):
    with seeded(seed):
        V0 = {'Y0', 'Z0', 'X0'}
        V1 = {'Y1', 'Z1', 'X1'}
        V2 = {'Y2', 'Z2', 'X2'}

        G = CD(V0 | V1 | V2,
               [('X0', 'Z0'), ('Z0', 'Y0'),
                ('X1', 'Z1'), ('Z1', 'Y1'),
                ('X2', 'Z2'), ('Z2', 'Y2'),
                ('X0', 'X1'), ('X1', 'X2')],
               [('X0', 'Y0', 'U_X0Y0'),
                ('X1', 'Y1', 'U_X1Y1'),
                ('X2', 'Y2', 'U_X2Y2')])

        # parametrization for U
        if devised:
            mu1 = {
                'U_X0Y0': 0.47, # emph Xt -> Yt
                'U_X1Y1': 0.55,
                'U_X2Y2': 0.51,

                'U_Y0': 0.02,  # denois Y0
                'U_Y1': 0.01,
                'U_Y2': 0.05,

                'U_Z0': 0.14,  # Normal Zt
                'U_Z1': 0.14,
                'U_Z2': 0.13,

                'U_X0': 0.85,
                'U_X1': 0.80,
                'U_X2' : 0.01
            }
        else:
            mu1 = {'U_X0Y0': rand_bw(0.01, 0.99, precision=2),
                   'U_X1Y1': rand_bw(0.01, 0.99, precision=2),
                   'U_X2Y2': rand_bw(0.01, 0.99, precision=2),
                   'U_X0': rand_bw(0.01, 0.99, precision=2),
                   'U_Y0': rand_bw(0.01, 0.99, precision=2),
                   'U_Z0': rand_bw(0.01, 0.99, precision=2),
                   'U_X1': rand_bw(0.01, 0.99, precision=2),
                   'U_Y1': rand_bw(0.01, 0.99, precision=2),
                   'U_Z1': rand_bw(0.01, 0.99, precision=2),
                   'U_Y2': rand_bw(0.01, 0.99, precision=2),
                   'U_Z2': rand_bw(0.01, 0.99, precision=2),
                   }

        domains = defaultdict(lambda: (0, 1))

        # SCM with parametrization
        M = StructuralCausalModel(G,
                                  F={
                                      'X0': lambda v: v['U_X0'] ^ v['U_X0Y0'],
                                      'Z0': lambda v: v['X0'] ^ v['U_Z0'],
                                      'Y0': lambda v: v['Z0'] ^ v['U_Y0'] ^ v['U_X0Y0'],

                                      'X1': lambda v: v['U_X1'] ^ v['U_X1Y1'] ^ v['X0'],
                                      'Z1': lambda v: v['X1'] ^ v['U_Z1'],
                                      'Y1': lambda v: v['Z1'] ^ v['U_Y1'] ^ v['U_X1Y1'],

                                      'X2': lambda v: v['U_X2'] ^ v['U_X2Y2'] ^ v['X1'],
                                      'Z2': lambda v: v['X2'] ^ v['U_Z2'],
                                      'Y2': lambda v: v['Z2'] ^ v['U_Y2'] ^ v['U_X2Y2'],
                                  },
                                  P_U=default_P_U(mu1),
                                  D=domains,
                                  more_U={'U_X0', 'U_Y0', 'U_Z0', 'U_X1', 'U_Z1', 'U_Y1', 'U_Z2', 'U_X2', 'U_Y2'})
        return M, mu1


# new graph for testing DUC based POMIS+
def W0toY2(devised=True, seed=None):
    with seeded(seed):
        V0 = {'Y0', 'X0', 'W0'}
        V1 = {'Y1', 'X1', 'W1'}
        V2 = {'Y2', 'X2', 'W2'}

        G = CD(V0 | V1 | V2,
               [('W0', 'X0'), ('X0', 'Y0'),
                            ('W1', 'X1'), ('X1', 'Y1'),
                            ('W2', 'X2'), ('X2', 'Y2'),

                            ('X0', 'X1'), ('X1', 'X2'),
                ],
               [
                ('X0', 'Y0', 'U_X0Y0'),
                ('X1', 'Y1', 'U_X1Y1'),
                ('X2', 'Y2', 'U_X2Y2'),

                ('X1', 'X2', 'U_X1X2'),
               ]
               )

        # parametrization for U
        if devised:
            mu1 = {
                'U_X0Y0': 0.52,
                'U_X1Y1': 0.44,
                'U_X2Y2': 0.48,

                'U_X1X2': 0.43,

                'U_Y0': 0.02,
                'U_Y1': 0.01,
                'U_Y2': 0.05,

                'U_X0': 0.82,
                'U_X1': 0.92,
                'U_X2' : 0.20,

                'U_W0': 0.15,
                'U_W1': 0.42,
                'U_W2': 0.41
            }
        else:
            mu1 = {'U_X0Y0': rand_bw(0.01, 0.99, precision=2),
                   'U_X1Y1': rand_bw(0.01, 0.99, precision=2),
                   'U_X2Y2': rand_bw(0.01, 0.99, precision=2),
                   'U_X0': rand_bw(0.01, 0.99, precision=2),
                   'U_Y0': rand_bw(0.01, 0.99, precision=2),
                   'U_Z0': rand_bw(0.01, 0.99, precision=2),
                   'U_X1': rand_bw(0.01, 0.99, precision=2),
                   'U_Y1': rand_bw(0.01, 0.99, precision=2),
                   'U_Z1': rand_bw(0.01, 0.99, precision=2),
                   'U_Y2': rand_bw(0.01, 0.99, precision=2),
                   'U_Z2': rand_bw(0.01, 0.99, precision=2),
                   }

        domains = defaultdict(lambda: (0, 1))

        # SCM with parametrization
        M = StructuralCausalModel(G,
                                  F={
                                      'W0': lambda v: v['U_W0'],
                                      'X0': lambda v: v['U_X0'] ^ v['W0']^ v['U_X0Y0'],
                                      'Y0': lambda v: v['X0'] ^ v['U_Y0'] ^ v['U_X0Y0'],

                                      'W1': lambda v: v['U_W1'],
                                      'X1': lambda v: v['U_X1'] ^ v['W1']^ v['U_X1X2']^ v['U_X1Y1'] ^ v['X0'],
                                      'Y1': lambda v: v['X1'] ^ v['U_Y1'] ^ v['U_X1Y1'],

                                      'W2': lambda v: v['U_W2'],
                                      'X2': lambda v: v['U_X2'] ^ v['W2'] ^ v['U_X1X2'] ^ v['U_X2Y2'] ^ v['X1'],
                                      'Y2': lambda v: v['X2'] ^ v['U_Y2'] ^ v['U_X2Y2'],
                                  },
                                  P_U=default_P_U(mu1),
                                  D=domains,
                                  more_U={
                                      'U_Y0',
                                      'U_Y1',
                                      'U_Y2',

                                      'U_X0', 'U_X1', 'U_X2',

                                      'U_W0', 'U_W1', 'U_W2',

                                      'U_X1X2',

                                      'U_X0Y0', 'U_X1Y1', 'U_X2Y2'
                                  })
        return M, mu1


# fig 3
def WttoYtprime(devised=True, seed=None):
    with seeded(seed):
        V0 = {'Y0', 'Z0', 'X0', 'W0'}
        V1 = {'Y1', 'Z1', 'X1', 'W1'}

        G = CD(V0 | V1,
               [
                   ('W0', 'X0'), ('X0', 'Z0'), ('Z0', 'Y0'),
                    ('W1', 'X1'), ('X1', 'Z1'), ('Z1', 'Y1'),
                    ('X0', 'X1')
                ],
               [
                ('X0', 'Y0', 'U_X0Y0'),
                ('X1', 'Y1', 'U_X1Y1'),]
               )

        # parametrization for U
        if devised:
            mu1 = {
                'U_X0Y0': 0.47,
                'U_X1Y1': 0.55,

                'U_Y0': 0.02,
                'U_Y1': 0.01,

                'U_X0': 0.15,
                'U_X1': 0.02,

                'U_Z0': 0.15,
                'U_Z1': 0.12,

                'U_W0': 0.15,
                'U_W1': 0.12,
            }
        else:
            mu1 = {'U_X0Y0': rand_bw(0.01, 0.99, precision=2),
                   'U_X1Y1': rand_bw(0.01, 0.99, precision=2),
                   'U_X2Y2': rand_bw(0.01, 0.99, precision=2),
                   'U_X0': rand_bw(0.01, 0.99, precision=2),
                   'U_Y0': rand_bw(0.01, 0.99, precision=2),
                   'U_Z0': rand_bw(0.01, 0.99, precision=2),
                   'U_X1': rand_bw(0.01, 0.99, precision=2),
                   'U_Y1': rand_bw(0.01, 0.99, precision=2),
                   'U_Z1': rand_bw(0.01, 0.99, precision=2),
                   'U_Y2': rand_bw(0.01, 0.99, precision=2),
                   'U_Z2': rand_bw(0.01, 0.99, precision=2),
                   }

        domains = defaultdict(lambda: (0, 1))

        # SCM with parametrization
        M = StructuralCausalModel(G,
                                  F={
                                      'W0': lambda v: v['U_W0'],
                                      'X0': lambda v: v['U_X0'] ^ v['W0']^ v['U_X0Y0'],
                                      'Z0': lambda v: v['U_Z0'] ^ v['X0'],
                                      'Y0': lambda v: v['Z0'] ^ v['U_Y0'] ^ v['U_X0Y0'],

                                      'W1': lambda v: v['U_W1'],
                                      'X1': lambda v: v['U_X1'] ^ v['W1']^ v['U_X1Y1'] ^ v['X0'],
                                      'Z1': lambda v: v['U_Z1'] ^ v['X1'],
                                      'Y1': lambda v: v['Z1'] ^ v['U_Y1'] ^ v['U_X1Y1']
                                  },
                                  P_U=default_P_U(mu1),
                                  D=domains,
                                  more_U={
                                      'U_Y0', 'U_Y1',

                                      'U_X0', 'U_X1',

                                      'U_Z0', 'U_Z1',

                                      'U_W0', 'U_W1',

                                      'U_X0Y0', 'U_X1Y1'
                                  })
        return M, mu1


if __name__ == '__main__':
    M, mu1 =W0toY2()
    G = M.G
    Vs = group_by_time_index(G.V)
    Ys = ['Y0', 'Y1', 'Y2']
    re = POMISplusSEQ(G=G, Vs=Vs, Ys=list(Ys), T=len(Ys) - 1)
    print(len(re))
