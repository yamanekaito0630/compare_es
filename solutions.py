# ライブラリのインポート
from mpi4py import MPI
from es.operator import Operator

import cv2
import gymnasium as gym
import logging
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_num_threads(1)


class BaseTorchSolution:
    def __init__(self, device):
        self.modules_to_learn = []
        self.device = torch.device(device)

    def get_action(self, obs):
        with torch.no_grad():
            return self._get_action(obs)

    def get_params(self):
        params = []
        with torch.no_grad():
            for layer in self.modules_to_learn:
                for p in layer.parameters():
                    params.append(p.cpu().numpy().ravel())
        return np.concatenate(params)

    def set_params(self, params):
        params = np.array(params)
        assert isinstance(params, np.ndarray)
        ss = 0
        for layer in self.modules_to_learn:
            for p in layer.parameters():
                ee = ss + np.prod(p.shape)
                p.data = torch.from_numpy(
                    params[ss:ee].reshape(p.shape)
                ).float().to(self.device)
                ss = ee
        assert ss == params.size

    def save(self, filename):
        params = self.get_params()
        np.savez(filename, params=params)

    def load(self, filename):
        with np.load(filename) as data:
            params = data['params']
            self.set_params(params)

    def get_num_params(self):
        return self.get_params().size

    def _get_action(self, obs):
        raise NotImplementedError()

    def reset(self):
        pass

    def get_fitness(self, worker_id, params, seed, num_rollouts):
        self.set_params(params)
        scores = []
        for _ in range(num_rollouts):
            obs, info = self.env.reset(seed=seed)
            done = False
            score = 0
            while not done:
                action = self.get_action(obs=obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                score += reward
                if terminated or truncated:
                    done = True

            scores.append(score)
        return np.mean(scores)

    @staticmethod
    def action_scale(actions):
        scaled_actions = []
        for action in actions:
            scaled_action = np.tanh(action)
            scaled_actions.append(scaled_action)

        return np.array(scaled_actions)

    @staticmethod
    def save_params(solver, solution, model_path):
        solution.set_params(solver.best_param())
        solution.save(model_path)

    def create_logger(self, name, log_dir=None, debug=False):
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            os.makedirs(log_dir + '/gen_es')
        log_format = '%(asctime)s %(process)d [%(levelname)s] %(message)s'
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format=log_format)
        logger = logging.getLogger(name)
        if log_dir:
            if not os.path.isfile(log_dir + '/hyper_parameters.txt'):
                os.path.join(log_dir, 'hyper_parameters.txt')
                with open(file=log_dir + '/hyper_parameters.txt', mode='a') as f:
                    f.write(
                        'env_name=' + self.env_name
                        + '\nnum_param=' + str(self.get_num_params())
                        + '\nalgolithm=' + str(self.algo_number)
                        + '\npopulation_size=' + str(self.popsize)
                        + '\nmax_iter=' + str(self.max_iter)
                        + '\nroll_out=' + str(self.reps)
                    )

            log_file = os.path.join(log_dir, '{}.txt'.format(name))
            file_hdl = logging.FileHandler(log_file)
            formatter = logging.Formatter(fmt=log_format)
            file_hdl.setFormatter(formatter)
            logger.addHandler(file_hdl)
        return logger

    @staticmethod
    def create_sigma_logger(log_dir, n_iter, sigma):
        if not os.path.isfile(log_dir + '/step_size.txt'):
            os.path.join(log_dir, 'step_size.txt')
        with open(file=log_dir + '/step_size.txt', mode='a') as f:
            f.write('Iter={0},{1:.5f}\n'.format(n_iter + 1, sigma))

    def train(self,
              t: int = 1,
              algo_number: int = 0,
              max_iter: int = 1000,
              reps: int = 1,
              is_resume: bool = False,
              from_iter: int = 1,
              log_dir: str = None,
              save_interval: int = 10,
              seed: int = 42,
              init_sigma: float = 0.1,
              init_best: float = -float('Inf'),
              ):
        ii32 = np.iinfo(np.int32)
        rnd = np.random.RandomState(seed=seed)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            self.algo_number = algo_number
            self.popsize = size * t
            self.max_iter = max_iter
            self.reps = reps
            self.seed = seed
            logger = self.create_logger(name='train_log', log_dir=log_dir)
            best_so_far = init_best

            num_params = self.get_num_params()
            print('#params={}'.format(num_params))
            init_params = num_params * [0]
            if is_resume:
                print("is_resume: True")
                solver = pickle.load(open(log_dir + '/gen_es/es_{}.pkl'.format(from_iter - 1), 'rb'))
            else:
                print("is_resume: False")
                from_iter = 1
                solver = Operator(dim=num_params,
                                  popsize=size * t,
                                  sigma0=init_sigma,
                                  x0=init_params).get_solver(algo_number=algo_number)
        comm.barrier()
        for n_iter in range(from_iter - 1, max_iter):
            task_seed = rnd.randint(0, ii32.max)
            comm.barrier()
            if rank == 0:
                self.create_sigma_logger(log_dir=log_dir, n_iter=n_iter, sigma=solver.get_sigma())
                params_sets = solver.ask()
                c_rank = 0
                for i in range(0, len(params_sets), t):
                    if c_rank == 0:
                        params_set = params_sets[i:i + t]
                    else:
                        data = params_sets[i:i + t]
                        comm.send(data, dest=c_rank, tag=c_rank)
                    c_rank += 1
            else:
                data = comm.recv(source=0, tag=rank)
                params_set = data

            fitness = []
            for i in range(t):
                f = self.get_fitness(rank, params_set[i], task_seed, reps)
                fitness.append(f)

            fitnesses = comm.gather(fitness, root=0)
            if rank == 0:
                fitnesses = np.concatenate(np.array(fitnesses))
                solver.tell(fitnesses)

                # ESの保存
                if (n_iter + 1) % save_interval == 0:
                    pickle.dump(solver, open(log_dir + '/gen_es/es_{}.pkl'.format(n_iter + 1), 'wb'))

                logger.info(
                    'Iter={0}, '
                    'max={1:.2f}, avg={2:.2f}, min={3:.2f}, std={4:.2f}'.format(
                        n_iter + 1, np.max(fitnesses), np.mean(fitnesses), np.min(fitnesses), np.std(fitnesses)))

                best_fitness = max(fitnesses)
                if best_fitness > best_so_far:
                    best_so_far = best_fitness
                    model_path = os.path.join(log_dir, 'best.npz')
                    self.save_params(solver=solver, solution=self, model_path=model_path)
                    logger.info('Best model updated, score={}'.format(best_fitness))
                if (n_iter + 1) % save_interval == 0:
                    model_path = os.path.join(log_dir, 'Iter_{}.npz'.format(n_iter + 1))
                    self.save_params(solver=solver, solution=self, model_path=model_path)
            self.reset()
        self.env.close()


class FCSolution(BaseTorchSolution):
    def __init__(self,
                 device,
                 env_name,
                 num_hidden_layers,
                 hidden_dim):
        super(FCSolution, self).__init__(device)

        self.env_name = env_name
        self.env = gym.make(env_name)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.extend([
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Tanh(),
            ])
        self.net = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=hidden_dim),
            *hidden_layers,
            nn.Linear(in_features=hidden_dim, out_features=act_dim),
        )
        self.modules_to_learn.append(self.net)

    def _get_action(self, obs):
        x = torch.tensor(obs)
        action = self.net(x)
        return action


class DCNN(BaseTorchSolution):
    def __init__(self,
                 device,
                 env_name):
        super(DCNN, self).__init__(device)

        self.env_name = env_name
        self.env = gym.make(env_name)
        obs_dim = self.env.observation_space.shape[2]
        act_dim = self.env.action_space.shape[0]
        self.prev_x = torch.zeros(self.env.observation_space.shape)

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=obs_dim * 2,
                out_channels=32,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=act_dim, dtype=torch.float32),
            nn.Tanh(),
        )
        self.modules_to_learn.append(self.cnn)

    def _get_action(self, obs):
        x = torch.tensor(obs)
        self.x_arg = torch.cat((x, self.prev_x), dim=2).unsqueeze(0) / 255
        self.prev_x = x
        self.x_arg = self.x_arg.permute(0, 3, 1, 2)
        self.output = self.cnn(self.x_arg)
        return self.output.squeeze(0).cpu().numpy()

    def reset(self):
        self.prev_x = torch.zeros(self.env.observation_space.shape)

    @staticmethod
    def to_heatmap(x):
        x = (x * 255).reshape(-1)
        cm = plt.get_cmap('jet')
        x = np.array([cm(int(np.round(xi)))[:3] for xi in x])
        return (x.reshape(224, 224, 3) * 255).astype(np.uint8)

    def show_gui(self, img, target: int):
        feat = self.cnn[:4](self.x_arg)
        feat = feat.clone().detach().requires_grad_(True)
        output = self.cnn[4:](feat)
        output[0][target].backward()
        alpha = torch.mean(feat.grad.view(32, 22 * 22), 1)
        feat = feat.view(32, 22, 22)
        L = F.relu(torch.sum(feat * alpha.view(-1, 1, 1), 0)).cpu().detach().numpy()
        L_min = np.min(L)
        L_max = np.max(L - L_min)
        L = (L - L_min) / L_max
        L = self.to_heatmap(cv2.resize(L, (224, 224)))
        img = cv2.resize(img, (224, 224))[:, :, ::-1]
        blended = img * 0.7 + L * 0.3
        cv2.imshow('render', blended.astype(np.uint8))
        cv2.waitKey(1)
