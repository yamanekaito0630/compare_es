import numpy as np
import math


def compute_ranks(x):
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


def compute_utilities(fitnesses):
    L = len(fitnesses)
    ranks = np.zeros_like(fitnesses)
    l = list(zip(fitnesses, range(L)))
    l.sort()
    for i, (_, j) in enumerate(l):
        ranks[j] = i
    # smooth reshaping
    utilities = np.array([max(0., x) for x in np.log(L / 2. + 1.0) - np.log(L - np.array(ranks))])
    utilities /= sum(utilities)  # make the utilities sum to 1
    utilities -= 1. / L  # baseline
    return utilities


def get_h_inv(dim: int) -> float:
    f = lambda a, b: ((1. + a * a) * exp(a * a / 2.) / 0.24) - 10. - dim
    f_prime = lambda a: (1. / 0.24) * a * exp(a * a / 2.) * (3. + a * a)
    h_inv = 1.0
    while abs(f(h_inv, dim)) > 1e-10:
        h_inv = h_inv - 0.5 * (f(h_inv, dim) / f_prime(h_inv))
    return h_inv


def exp(a: float) -> float:
    return math.exp(min(100, a))  # avoid overflow


def sort_indices_by(evals, z):
    lam = len(evals)
    sorted_indices = np.argsort(evals)
    sorted_evals = evals[sorted_indices]
    no_of_feasible_solutions = np.where(sorted_evals != np.inf)[0].size
    if no_of_feasible_solutions != lam:
        infeasible_z = z[:, np.where(evals == np.inf)[0]]
        distances = np.sum(infeasible_z ** 2, axis=0)
        infeasible_indices = sorted_indices[no_of_feasible_solutions:]
        indices_sorted_by_distance = np.argsort(distances)
        sorted_indices = sorted_indices.at[no_of_feasible_solutions:].set(
            infeasible_indices[indices_sorted_by_distance])
    return sorted_indices


class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.dim = pi.num_params
        self.epsilon = epsilon
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.pi.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class BasicSGD(Optimizer):
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def _compute_step(self, globalg):
        step = -self.stepsize * globalg
        return step


class SGD(Optimizer):
    def __init__(self, pi, stepsize, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


class CMAES:
    """CMA-ES wrapper."""

    def __init__(self, num_params,  # number of model parameters
                 init_params,
                 sigma_init=0.10,  # initial standard deviation
                 popsize=255,  # population size
                 weight_decay=0.01):  # weight decay coefficient

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.solutions = None

        import cma
        self.es = cma.CMAEvolutionStrategy(init_params,
                                           self.sigma_init,
                                           {'popsize': self.popsize,
                                            })

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        """returns a list of parameters"""
        self.solutions = np.array(self.es.ask())
        return self.solutions

    def tell(self, reward_table_result):
        reward_table = np.array(reward_table_result)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay
        self.es.tell(self.solutions, (-reward_table).tolist())  # convert minimizer to maximizer.

    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        r = self.es.result
        return r[0], -r[1], -r[1], r[6]

    def get_sigma(self):
        return np.mean(self.es.result[6])


class SimpleGA:
    """Simple Genetic Algorithm."""

    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.1,  # initial standard deviation
                 sigma_decay=0.999,  # anneal standard deviation
                 sigma_limit=0.01,  # stop annealing if less than this
                 popsize=256,  # population size
                 elite_ratio=0.1,  # percentage of the elites
                 forget_best=False,  # forget the historical best elites
                 weight_decay=0.01,  # weight decay coefficient
                 ):

        self.epsilon = None
        self.solutions = None
        self.curr_best_reward = None
        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.popsize = popsize

        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

        self.sigma = self.sigma_init
        self.elite_params = np.zeros((self.elite_popsize, self.num_params))
        self.elite_rewards = np.zeros(self.elite_popsize)
        self.best_param = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_iteration = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay

    def rms_stdev(self):
        return self.sigma  # same sigma for all parameters.

    def ask(self):
        """returns a list of parameters"""
        self.epsilon = np.random.randn(self.popsize, self.num_params) * self.sigma
        solutions = []

        def mate(a, b):
            c = np.copy(a)
            idx = np.where(np.random.rand(c.size) > 0.5)
            c[idx] = b[idx]
            return c

        elite_range = range(self.elite_popsize)
        for i in range(self.popsize):
            idx_a = np.random.choice(elite_range)
            idx_b = np.random.choice(elite_range)
            child_params = mate(self.elite_params[idx_a], self.elite_params[idx_b])
            solutions.append(child_params + self.epsilon[i])

        solutions = np.array(solutions)
        self.solutions = solutions

        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        if self.forget_best or self.first_iteration:
            reward = reward_table
            solution = self.solutions
        else:
            reward = np.concatenate([reward_table, self.elite_rewards])
            solution = np.concatenate([self.solutions, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_popsize]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]

        if self.first_iteration or (self.curr_best_reward > self.best_reward):
            self.first_iteration = False
            self.best_reward = self.elite_rewards[0]
            self.best_param = np.copy(self.elite_params[0])

        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

    def current_param(self):
        return self.elite_params[0]

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.best_param

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return self.best_param, self.best_reward, self.curr_best_reward, self.sigma

    def get_sigma(self):
        return np.mean(self.sigma)


class OpenES:
    """ Basic Version of OpenAI Evolution Strategies."""

    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.1,  # initial standard deviation
                 sigma_decay=0.999,  # anneal standard deviation
                 sigma_limit=0.01,  # stop annealing if less than this
                 learning_rate=0.01,  # learning rate for standard deviation
                 learning_rate_decay=0.9999,  # annealing the learning rate
                 learning_rate_limit=0.001,  # stop annealing learning rate
                 popsize=256,  # population size
                 antithetic=False,  # whether to use antithetic sampling
                 weight_decay=0.01,  # weight decay coefficient
                 rank_fitness=True,  # use rank rather than fitness numbers
                 forget_best=True):  # forget historical best

        self.epsilon_half = None
        self.epsilon = None
        self.solutions = None
        self.curr_best_reward = None
        self.curr_best_mu = None
        self.num_params = num_params
        self.sigma_decay = sigma_decay
        self.sigma = sigma_init
        self.sigma_init = sigma_init
        self.sigma_limit = sigma_limit
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.popsize / 2)

        self.reward = np.zeros(self.popsize)
        self.mu = np.zeros(self.num_params)
        self.best_mu = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_interation = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = Adam(self, learning_rate)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        """returns a list of parameters"""
        # antithetic sampling
        if self.antithetic:
            self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
            self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])
        else:
            self.epsilon = np.random.randn(self.popsize, self.num_params)

        self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

        return self.solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward = np.array(reward_table_result)

        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward += l2_decay

        idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        best_mu = self.solutions[idx[0]]

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # main bit:
        # standardize the rewards to have a gaussian distribution
        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        change_mu = 1. / (self.popsize * self.sigma) * np.dot(self.epsilon.T, normalized_reward)

        # self.mu += self.learning_rate * change_mu

        self.optimizer.stepsize = self.learning_rate
        self.optimizer.update(-change_mu)

        # adjust sigma according to the adaptive sigma calculation
        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

        if self.learning_rate > self.learning_rate_limit:
            self.learning_rate *= self.learning_rate_decay

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return self.best_mu, self.best_reward, self.curr_best_reward, self.sigma

    def get_sigma(self):
        return np.mean(self.sigma)


class PEPG:
    """Extension of PEPG with bells and whistles."""

    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.10,  # initial standard deviation
                 sigma_alpha=0.20,  # learning rate for standard deviation
                 sigma_decay=0.999,  # anneal standard deviation
                 sigma_limit=0.01,  # stop annealing if less than this
                 sigma_max_change=0.2,  # clips adaptive sigma to 20%
                 learning_rate=0.01,  # learning rate for standard deviation
                 learning_rate_decay=0.9999,  # annealing the learning rate
                 learning_rate_limit=0.01,  # stop annealing learning rate
                 elite_ratio=0,  # if > 0, then ignore learning_rate
                 popsize=256,  # population size
                 average_baseline=True,  # set baseline to average of batch
                 weight_decay=0.01,  # weight decay coefficient
                 rank_fitness=True,  # use rank rather than fitness numbers
                 forget_best=True):  # don't keep the historical best solution

        self.solutions = None
        self.epsilon_full = None
        self.epsilon = None
        self.curr_best_reward = None
        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.sigma_max_change = sigma_max_change
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.average_baseline = average_baseline
        if self.average_baseline:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.batch_size = int(self.popsize / 2)
        else:
            assert (self.popsize & 1), "Population size must be odd"
            self.batch_size = int((self.popsize - 1) / 2)

        # option to use greedy es method to select next mu, rather than using drift param
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.use_elite = False
        if self.elite_popsize > 0:
            self.use_elite = True

        self.forget_best = forget_best
        self.batch_reward = np.zeros(self.batch_size * 2)
        self.mu = np.zeros(self.num_params)
        self.sigma = np.ones(self.num_params) * self.sigma_init
        self.curr_best_mu = np.zeros(self.num_params)
        self.best_mu = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_interation = True
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = Adam(self, learning_rate)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        """returns a list of parameters"""
        # antithetic sampling
        self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
        self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
        if self.average_baseline:
            epsilon = self.epsilon_full
        else:
            # first population is mu, then positive epsilon, then negative epsilon
            epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
        solutions = self.mu.reshape(1, self.num_params) + epsilon
        self.solutions = solutions
        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.rank_fitness:
            reward_table = compute_centered_ranks(reward_table)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        reward_offset = 1
        if self.average_baseline:
            b = np.mean(reward_table)
            reward_offset = 0
        else:
            b = reward_table[0]  # baseline

        reward = reward_table[reward_offset:]
        if self.use_elite:
            idx = np.argsort(reward)[::-1][0:self.elite_popsize]
        else:
            idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        if best_reward > b or self.average_baseline:
            best_mu = self.mu + self.epsilon_full[idx[0]]
            best_reward = reward[idx[0]]
        else:
            best_mu = self.mu
            best_reward = b

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.sigma = np.ones(self.num_params) * self.sigma_init
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # short hand
        epsilon = self.epsilon
        sigma = self.sigma

        # update the mean

        # move mean to the average of the best idx means
        if self.use_elite:
            self.mu += self.epsilon_full[idx].mean(axis=0)
        else:
            r_t = (reward[:self.batch_size] - reward[self.batch_size:])
            change_mu = np.dot(r_t, epsilon)
            self.optimizer.stepsize = self.learning_rate
            self.optimizer.update(-change_mu)
            # self.mu += (change_mu * self.learning_rate) # normal SGD method

        # adaptive sigma
        # normalization
        if self.sigma_alpha > 0:
            stdev_reward = 1.0
            if not self.rank_fitness:
                stdev_reward = reward.std()
            s = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
            reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
            r_s = reward_avg - b
            delta_sigma = (np.dot(r_s, s)) / (2 * self.batch_size * stdev_reward)

            # adjust sigma according to the adaptive sigma calculation
            # for stability, don't let sigma move more than 10% of orig value
            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
            change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
            self.sigma += change_sigma

        if self.sigma_decay < 1:
            self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

        if self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit:
            self.learning_rate *= self.learning_rate_decay

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return self.best_mu, self.best_reward, self.curr_best_reward, self.sigma

    def get_sigma(self):
        return np.mean(self.sigma)


class SNES:
    """ Separable NES """

    def __init__(self, x0, popsize, sigma_init):
        self.x0 = x0
        self.batchSize = popsize
        self.dim = len(x0)
        self.learningRate = 0.45 * (10 + np.log(self.dim)) / np.sqrt(self.dim)
        # print self.learningRate
        # self.learningRate = self.learningRate * learning_rate_mult
        # self.learningRate = 0.000001
        self.numEvals = 0
        self.bestFound = None
        self.sigmas = np.ones(self.dim) * sigma_init
        self.bestFitness = -np.Inf
        self.center = x0.copy()

    def ask(self):
        self.samples = [np.random.randn(self.dim) for _ in range(self.batchSize)]
        asked = [(self.sigmas * s + self.center) for s in self.samples]
        self.asked = asked
        return asked

    def tell(self, fitnesses):
        samples = self.samples
        if max(fitnesses) > self.bestFitness:
            self.bestFitness = max(fitnesses)
            self.bestFound = samples[np.argmax(fitnesses)]
        self.numEvals += self.batchSize

        # update center and variances
        utilities = compute_utilities(fitnesses)
        self.center += self.sigmas * np.dot(utilities, samples)
        cov_gradient = np.dot(utilities, [s ** 2 - 1 for s in samples])
        self.sigmas = self.sigmas * np.exp(0.5 * self.learningRate * cov_gradient)

    def best_param(self):
        return self.bestFound

    def get_sigma(self):
        return np.mean(self.sigmas)


class CRFMNES:
    """ CR-FM-NES """

    def __init__(self, num_dims: int, popsize: int, mean_value, init_sigma: float):
        if popsize % 2 == 1:
            popsize += 1
        self.lamb = popsize
        self.dim = num_dims
        self.sigma = init_sigma
        mean = np.ones([num_dims, 1]) * mean_value
        self.m = np.array([mean]).T
        self.v = np.random.randn(num_dims, 1) / np.sqrt(num_dims)
        self.D = np.ones([num_dims, 1])

        self.w_rank_hat = (np.log(self.lamb / 2 + 1) - np.log(np.arange(1, self.lamb + 1))).reshape(self.lamb, 1)
        self.w_rank_hat[np.where(self.w_rank_hat < 0)] = 0
        self.w_rank = self.w_rank_hat / sum(self.w_rank_hat) - (1. / self.lamb)
        self.mueff = np.dot(1 / ((self.w_rank + (1 / self.lamb)).T, (self.w_rank + (1 / self.lamb)))[0][0])

        self.cs = (self.mueff + 2.) / (self.dim + self.mueff + 5.)
        self.cc = (4. + self.mueff / self.dim) / (self.dim + 4. + 2. * self.mueff / self.dim)
        self.c1_cma = 2. / (math.pow(self.dim + 1.3, 2) + self.mueff)
        # initialization
        self.chiN = math.sqrt(self.dim) * (1. - 1. / (4. * self.dim) + 1. / (21. * self.dim * self.dim))
        self.pc = np.zeros((self.dim, 1))
        self.ps = np.zeros((self.dim, 1))
        # distance weight parameter
        self.h_inv = get_h_inv(self.dim)
        self.alpha_dist = lambda lambF: self.h_inv * min(1., math.sqrt(self.lamb / self.dim)) * math.sqrt(
            lambF / self.lamb)
        self.w_dist_hat = lambda z, lambF: exp(self.alpha_dist(lambF) * np.linalg.norm(z))
        # learning rate
        self.eta_m = 1.0
        self.eta_move_sigma = 1.
        self.eta_stag_sigma = lambda lambF: math.tanh((0.024 * lambF + 0.7 * self.dim + 20.) / (self.dim + 12.))
        self.eta_conv_sigma = lambda lambF: 2. * math.tanh((0.025 * lambF + 0.75 * self.dim + 10.) / (self.dim + 4.))
        self.c1 = lambda lambF: self.c1_cma * (self.dim - 5) / 6 * (lambF / self.lamb)
        self.eta_B = lambda lambF: np.tanh((min(0.02 * lambF, 3 * np.log(self.dim)) + 5) / (0.23 * self.dim + 25))

        self.g = 0
        self.no_of_evals = 0
        self.iteration = 0
        self.stop = 0

        self.idxp = np.arange(self.lamb / 2, dtype=int)
        self.idxm = np.arange(self.lamb / 2, self.lamb, dtype=int)
        self.z = np.zeros([self.dim, self.lamb])

        self.f_best = float('inf')
        self.x_best = np.empty(self.dim)

    def set_m(self, params):
        self.m = np.array(params).reshape((self.dim, 1))

    def ask(self):
        zhalf = np.random.randn(self.dim, int(self.lamb / 2))
        self.z = self.z.at[:, self.idxp].set(zhalf)
        self.z = self.z.at[:, self.idxm].set(-zhalf)
        self.normv = np.linalg.norm(self.v)
        self.normv2 = self.normv ** 2
        self.vbar = self.v / self.normv
        self.y = self.z + ((np.sqrt(1 + self.normv2) - 1) * np.dot(self.vbar, np.dot(self.vbar.T, self.z)))
        self.x = self.m + (self.sigma * self.y) * self.D
        return self.x.T

    def tell(self, evals_no_sort) -> None:
        sorted_indices = sort_indices_by(evals_no_sort, self.z)
        best_eval_id = sorted_indices[0]
        f_best = evals_no_sort[best_eval_id]
        x_best = self.x[:, best_eval_id]
        self.z = self.z[:, sorted_indices]
        y = self.y[:, sorted_indices]
        x = self.x[:, sorted_indices]
        self.no_of_evals += self.lamb
        self.g += 1
        if f_best < self.f_best:
            self.f_best = f_best
            self.x_best = x_best

            # This operation assumes that if the solution is infeasible, infinity comes in as input.
        lambF = np.sum(evals_no_sort < np.finfo(float).max)
        # evolution path p_sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2. - self.cs) * self.mueff) * np.dot(self.z, self.w_rank)
        ps_norm = np.linalg.norm(self.ps)
        # distance weight
        f1 = self.h_inv * min(1., math.sqrt(self.lamb / self.dim)) * math.sqrt(lambF / self.lamb)
        w_tmp = self.w_rank_hat * np.exp(np.linalg.norm(self.z, axis=0) * f1).reshape((self.lamb, 1))
        weights_dist = w_tmp / sum(w_tmp) - 1. / self.lamb
        # switching weights and learning rate
        weights = weights_dist if ps_norm >= self.chiN else self.w_rank
        eta_sigma = self.eta_move_sigma if ps_norm >= self.chiN else self.eta_stag_sigma(
            lambF) if ps_norm >= 0.1 * self.chiN else self.eta_conv_sigma(lambF)
        # update pc, m
        wxm = np.dot(x - self.m, weights)
        self.pc = (1. - self.cc) * self.pc + np.sqrt(self.cc * (2. - self.cc) * self.mueff) * wxm / self.sigma
        self.m += self.eta_m * wxm
        normv4 = self.normv2 ** 2
        exY = np.append(y, self.pc / self.D, axis=1)  # dim x lamb+1
        yy = exY * exY  # dim x lamb+1
        ip_yvbar = np.dot(self.vbar.T, exY)
        yvbar = exY * self.vbar  # dim x lamb+1. exYのそれぞれの列にvbarがかかる
        gammav = 1. + self.normv2
        vbarbar = self.vbar * self.vbar
        alphavd = min(
            [1, math.sqrt(normv4 + (2 * gammav - math.sqrt(gammav)) / np.max(vbarbar)) / (
                    2 + self.normv2)])  # scalar
        t = exY * ip_yvbar - self.vbar * (ip_yvbar ** 2 + gammav) / 2  # dim x lamb+1
        b = -(1 - alphavd ** 2) * normv4 / gammav + 2 * alphavd ** 2
        H = np.ones([self.dim, 1]) * 2 - (b + 2 * alphavd ** 2) * vbarbar  # dim x 1
        invH = H ** (-1)
        s_step1 = yy - self.normv2 / gammav * (yvbar * ip_yvbar) - np.ones([self.dim, self.lamb + 1])  # dim x lamb+1
        ip_vbart = np.dot(self.vbar.T, t)  # 1 x lamb+1
        s_step2 = s_step1 - alphavd / gammav * (
                (2 + self.normv2) * (t * self.vbar) - self.normv2 * np.dot(vbarbar, ip_vbart))  # dim x lamb+1
        invHvbarbar = invH * vbarbar
        ip_s_step2invHvbarbar = np.dot(invHvbarbar.T, s_step2)  # 1 x lamb+1
        div = 1 + b * np.dot(vbarbar.T, invHvbarbar)
        s = (s_step2 * invH) - b / div * np.dot(invHvbarbar, ip_s_step2invHvbarbar)  # dim x lamb+1
        ip_svbarbar = np.dot(vbarbar.T, s)  # 1 x lamb+1
        t = t - alphavd * ((2 + self.normv2) * (s * self.vbar) - np.dot(self.vbar, ip_svbarbar))  # dim x lamb+1
        # update v, D
        exw = np.append(self.eta_B(lambF) * weights, np.full((1, 1), self.c1(lambF)), axis=0)  # lamb+1 x 1
        self.v = self.v + np.dot(t, exw) / self.normv
        self.D = self.D + np.dot(s, exw) * self.D
        # calculate detA
        nthrootdetA = exp(
            np.sum(np.log(self.D)) / self.dim + np.log(1 + np.dot(self.v.T, self.v)[0][0]) / (2 * self.dim))
        self.D = self.D / nthrootdetA
        # update sigma
        G_s = np.sum(np.dot((self.z * self.z - np.ones([self.dim, self.lamb])), weights)) / self.dim
        self.sigma = self.sigma * exp(eta_sigma / 2 * G_s)

    def best_param(self):
        return self.set_m(self.x_best)

    def get_sigma(self):
        return self.sigma
