from itertools import product

from npsem.NIPS2025POMISPLUS_exp.scm_examples import X0toY2, W0toY2, WttoYtprime
from npsem.utils import combinations

if __name__ == '__main__':
    for name, (model, p_u) in [
        ('X0toY2', X0toY2(True, seed=0), 100000, {'Y0', 'Y1', 'Y2'}, True),
        ('W0toY2', W0toY2(True, seed=0), 100000, {'Y0', 'Y1', 'Y2'}, False),
        ('WttoYtprime', WttoYtprime(True, seed=0), 10000, {'Y0', 'Y1'}, False),
        ]:
        print('=========================================================================')
        print(f'========================={str(name).center(23)}=========================')
        print(p_u)
        for x_var in combinations(model.G.V - {'Y'}):
            for x_val in product(*[(0, 1) for x in x_var]):
                results = model.query(('Y',), intervention=dict(zip(x_var, x_val)))
                print(f'{str(dict(zip(x_var,x_val))).rjust(45)}:   {results[(1,)]:.2f} ({results[(1,)]})')
        print('=========================================================================')
        print('\n\n\n\n')
