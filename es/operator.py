import sys

from es.es import CMAES, SimpleGA, PEPG, OpenES, SNES, CRFMNES


class Operator:
    def __init__(self, dim, popsize, sigma0, x0):
        self.dim = dim
        self.popsize = popsize
        self.sigma0 = sigma0
        self.x0 = x0

    def get_solver(self, algo_number: int = 0):
        if algo_number == 0:
            solver = CMAES(
                num_params=self.dim,
                init_params=self.x0,
                sigma_init=self.sigma0,
                popsize=self.popsize,
                weight_decay=0.01
            )
        elif algo_number == 1:
            solver = SNES(
                x0=self.x0,
                popsize=self.popsize,
                sigma_init=self.sigma0
            )
        elif algo_number == 2:
            solver = SimpleGA(
                num_params=self.dim,
                sigma_init=self.sigma0,
                sigma_decay=0.999,
                sigma_limit=0.01,
                popsize=self.popsize,
                elite_ratio=0.1,
                forget_best=False,
                weight_decay=0.01,
            )
        elif algo_number == 3:
            solver = PEPG(
                num_params=self.dim,
                sigma_init=self.sigma0,
                sigma_alpha=0.20,
                sigma_decay=0.999,
                sigma_limit=0.01,
                sigma_max_change=0.2,
                learning_rate=0.01,
                learning_rate_decay=0.9999,
                learning_rate_limit=0.01,
                elite_ratio=0,
                popsize=self.popsize,
                average_baseline=True,
                weight_decay=0.01,
                rank_fitness=True,
                forget_best=True
            )
        elif algo_number == 4:
            solver = OpenES(
                num_params=self.dim,
                sigma_init=self.sigma0,
                sigma_decay=0.999,
                sigma_limit=0.01,
                learning_rate=0.01,
                learning_rate_decay=0.9999,
                learning_rate_limit=0.001,
                popsize=self.popsize,
                antithetic=False,
                weight_decay=0.01,
                rank_fitness=True,
                forget_best=True
            )
        elif algo_number == 5:
            solver = CRFMNES(
                num_dims=self.dim,
                popsize=self.popsize,
                mean_value=1,
                init_sigma=self.sigma0
            )
        else:
            print("algo_number should be selected between 0 to 5.")
            sys.exit()

        return solver
