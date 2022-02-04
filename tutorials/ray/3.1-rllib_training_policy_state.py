import ray
from ray import tune
import ray.rllib.agents.ppo as ppo

ray.init()
analysis = tune.run(
    "PPO",
    name="training_api",
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

checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")

last_checkpoint = analysis.get_last_checkpoint()
last_checkpoint = analysis.get_last_checkpoint(
    metric="episode_reward_mean", mode="max"
)

trainer = ppo.PPOTrainer()
trainer.restore(last_checkpoint)

default_policy_weights = trainer.get_policy().get_weights()
default_policy_weights = trainer.workers.local_worker().policy_map["default_policy"].get_weights()

list_weights_by_worker = trainer.workers.foreach_worker(lambda w: w.get_policy)
list_weights_by_worker = trainer.workers.foreach_worker_with_index(lambda w, i: w.get_policy().get_weights())
