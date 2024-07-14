from solutions import DCNN

import argparse
import torch
import gymnasium as gym


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', help='Name of environment.', type=str, default='CarRacing-v2')
    parser.add_argument('--load-model', help='Path to model', type=str, default='log/CarRacing-v2/algo_number_4/trial_1/best.npz')
    config, _ = parser.parse_known_args()
    return config

def main(config):
    device = torch.device('cpu')
    agent = DCNN(
        device=device,
        env_name=config.env_name
    )
    agent.load(config.load_model)

    env = gym.make(config.env_name, render_mode="human")
    observation, info = env.reset()
    for _ in range(1000):
        action = agent.get_action(observation)
        agent.show_gui(observation, 0)
        agent.show_gui(observation, 1)
        agent.show_gui(observation, 2)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
