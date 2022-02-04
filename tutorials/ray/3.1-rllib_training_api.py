import ray
from ray import tune
import ray.rllib.agents.ppo as ppo

ray.init()
analysis = tune.run(
    "PPO",
    config={
        "env": "CartPole-v0",
        "num_gpus": 1,
        "framework": "torch",
        "num_workers": 15,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    },
    stop={
        "episode_reward_mean": 200,
    }
)