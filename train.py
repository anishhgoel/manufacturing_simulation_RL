import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import ManufacturingEnv

def main():
    env = ManufacturingEnv()
    vec_env = DummyVecEnv([lambda: env])
    #hyperparametrs for PPO
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        n_steps=1024,                        # smaller rollout lenfth (default is 2048)
        policy_kwargs={"net_arch":[128,128]}, #custom architecture
        verbose = 1
    )

    print("Training the model..")
    model.learn(total_timesteps=50000)
    model.save("ppo_manufacturing")
    print("Model is saved as ppo_manufacturing.zip")

if __name__ == "__main__":
    main()
