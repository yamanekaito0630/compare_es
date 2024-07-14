import gymnasium as gym
import cv2

env = gym.make("CarRacing-v2", domain_randomize=False, render_mode="human")

obs_dim = env.observation_space.shape
act_dim = env.action_space.shape

print("obs_dim:", obs_dim)
print("act_dim:", act_dim)

# reset with no colour scheme change
observation, info = env.reset(options={"randomize": False})

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    img = cv2.resize(observation, (400, 400))[:, :, ::-1]
    cv2.imshow("render", img)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
