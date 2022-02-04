from gettext import translation
from pyexpat import model
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models.preprocessors import get_preprocessor
import gym


print("Get policy model: (pong game)")
env = gym.make("Pong-v0")

preprocessor = get_preprocessor(env.observation_space)(env.observation_space)

print(env.reset().shape)
print(preprocessor.transform(env.reset()).shape)

print("\n\n Example: Querying a policy’s action distribution (cart-pole game)")
trainer = PPOTrainer(env="CartPole-v0", config={"framework": "torch", "num_workers": 0)
policy = trainer.get_policy()

logits, _ = policy.model.from_batch({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})

print(policy.dist_classes)
dist = policy.dist_class(logits, policy.model)

print(dist.sample())
print(dist.logp([1]))

values = policy.model.value_function()
print(policy.model.base_model.summary())




print(Example: Getting Q values from a DQN model)
from ray.rllib.agents.dqn import DQNTrainer
trainer = DQNTrainer(env="CartPole-v0", config={"framework": "torch"})
model = trainer.get_policy().model
print(model.variables())
model_out = model.from_batch({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})
print(f"Base summary: {model.base_summary()}")

print(f"Q values (specific to dqn)\n: {model.get_q_value_distributions(model_out)}")
print(f"State value model (specific to DQN)\n: {model.get_state_value(model_out)}")
print(f"State value head: {model.state_value_head.summary()}")