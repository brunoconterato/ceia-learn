import gym
import ray
from ray import tune
from ray.rllib.agents.registry import get_trainer_class
import torch

if __name__ == "__main__":
    print(f'GPU available: {torch.cuda.is_available()}')

    ray.init(num_gpus=1, num_cpus=16)

    config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": "Taxi-v3",
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 14,
        "num_gpus": 1,
        "num_envs_per_worker": 1,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },

        # Size of batches collected from each worker.
        "rollout_fragment_length": 400,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 4000,

        # Set up a separate evaluation worker set for the
        # `trainer.evaluate()` call after training (see below).
        "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": False,
        }
    }

    stop = {
        "training_iteration": 50,
        # "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }

    print("Training policy until desired reward/timesteps/iterations. ...")
    results = tune.run(
        "PPO",
        name="inference_after_training",
        config=config,
        stop=stop,
        verbose=2,
        checkpoint_freq=1,
        checkpoint_at_end=True,
        keep_checkpoints_num=10,
    )

    print("Training completed. Restoring new Trainer for action inference.")
    checkpoint = results.get_last_checkpoint()
    # Create new Trainer and restore its state from the last checkpoint.
    trainer = get_trainer_class("PPO")(config=config)
    trainer.restore(checkpoint)

    # Create the env to do inference in.
    env = gym.make("Taxi-v3")
    obs = env.reset()
    
    num_episodes = 0
    episode_reward = 0.0

    while num_episodes < 10:
        action = trainer.compute_single_action(
            observation=obs,
            policy_id="default_policy",
        )
        obs, reward, done, info = env.step(action)
        # env.render()
        # input("Press Enter to continue...")
        episode_reward += reward

        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            env.reset()
            num_episodes += 1
            episode_reward = 0.0

        ray.shutdown()