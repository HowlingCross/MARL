import numpy as np
from stable_baselines3 import PPO
from cacc_ncars_env import CACCNCarsEnv

def main():
    print("Környezet és modell betöltése teszteléshez...")
    env = CACCNCarsEnv(num_cars=3)
    model = PPO.load("ppo_cacc_model")
    
    obs, infos = env.reset()
    total_rewards = {agent: 0.0 for agent in env.possible_agents}
    steps = 0
    crashed = False

    print("Szimuláció futtatása...")
    while env.agents:
        actions = {}
        for agent in env.agents:
            action, _states = model.predict(obs[agent], deterministic=True)
            actions[agent] = action
            
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
            if rewards[agent] <= -50.0:
                crashed = True
        steps += 1

    print("\n--- TESZT EREDMÉNYEK ---")
    print(f"Lefutott időlépések: {steps}")
    print(f"Ütközés történt: {'IGEN' if crashed else 'NEM'}")
    print("Összjutalom autónként:")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.2f}")

if __name__ == "__main__":
    main()
