# ライブラリのインポート
from solutions import FCSolution

import argparse
import torch
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', help='Num of loop', type=int, default=1)
    parser.add_argument('--algo-number', help='0: CMAES, 1: SNES, 2: SimpleGA, 3: PEPG, 4: OpenES, 5: CRFMNES', type=int, default=0)
    parser.add_argument('--trials', required=True, nargs="*", help='Num list of robo', type=int)
    parser.add_argument('--env-name', help='Name of environment.', type=str, default='default')
    parser.add_argument('--max-iter', help='Max training iterations.', type=int, default=1000)
    parser.add_argument('--is-resume', help='Restart training', type=int, default=0)
    parser.add_argument('--from-iter', help='From training iterations.', type=int, default=1)
    parser.add_argument('--save-interval', help='Model saving period.', type=int, default=50)
    parser.add_argument('--reps', help='Number of rollouts for fitness.', type=int, default=1)
    parser.add_argument('--init-sigma', help='Initial std.', type=float, default=0.1)
    parser.add_argument('--init-best', help='Initial best.', type=float, default=-float('Inf'))
    config, _ = parser.parse_known_args()
    return config


def main(config):
    device = torch.device('cpu')
    t_fs = []
    for trial_number in config.trials:
        log_dir = 'log/{}/algo_number_{}/trial_{}'.format(config.env_name, config.algo_number, trial_number)
        agent = FCSolution(
            device=device,
            env_name=config.env_name,
            num_hidden_layers=2,
            hidden_dim=32
        )

        t_f = agent.train(
            t=config.t,
            algo_number=config.algo_number,
            max_iter=config.max_iter,
            reps=config.reps,
            is_resume=config.is_resume,
            from_iter=config.from_iter,
            save_interval=config.save_interval,
            log_dir=log_dir,
            seed=trial_number,
            init_sigma=config.init_sigma,
            init_best=config.init_best,
        )
        t_fs.append(t_f)
        pickle.dump(t_fs, open('log/{}/algo_number_{}/fitness_transitions.pkl'.format(config.env_name, config.algo_number), 'wb'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
