import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import ManufacturingEnv

def visualize_process(episodes=2):
    model = PPO.load("ppo_manufacturing")
    
    env = ManufacturingEnv(max_steps=300)
    
    for episode in range(episodes):
        obs, _ = env.reset()
        temperatures = []
        qualities = []
        
        done = False
        truncated = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            temperatures.append(obs[0])
            qualities.append(obs[1])
            episode_reward += reward
            steps += 1
            
        print(f"Episode {episode+1} ended after {steps} steps with final quality={obs[1]:.2f}")
        
        # Plotting
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(temperatures, label='Temperature')
        plt.axhline(env.optimal_temp, color='r', linestyle='--', label='Optimal Temp')
        plt.title(f"Episode {episode+1} - Temperature")
        plt.xlabel("Timestep")
        plt.ylabel("Temperature")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(qualities, color='g', label='Quality')
        plt.axhline(100, color='r', linestyle='--', label='Max Quality')
        plt.axhline(0, color='y', linestyle='--', label='Min Quality')
        plt.title(f"Episode {episode+1} - Quality")
        plt.xlabel("Timestep")
        plt.ylabel("Quality")
        plt.legend()

        plt.suptitle(f"Episode Reward: {episode_reward:.2f}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_process(episodes=3)