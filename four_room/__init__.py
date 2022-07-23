import gym
from gym.envs.registration import register

register(
    id='Fourrooms-v0',
    entry_point='four_room.envs.fourrooms:Fourrooms',
    max_episode_steps=500,
)