import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

ray.init()
config = ppo.PPOTrainer.get_default_config()
config["num_gpus"] = 1
config["numworkers"] = 7
trainer = ppo.PPOTrainer(config=config, , env="CartPole-v0")


checkpoint = ""
for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))

    if i % 100 == 99:
        checkpoint = trainer.save()

    print(f"checkpoint saved at: {checkpoint}")


trainer.import_model(checkpoint)

