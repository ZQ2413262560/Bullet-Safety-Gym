import gym
import bullet_safety_gym

env = gym.make('SafetyAntGather-v0')

while True:
    done = False
    env.render()  # make GUI of PyBullet appear
    x = env.reset()
    while not done:
        random_action = env.action_space.sample()
        x, reward, done, info = env.step(random_action)
