import numpy as np
from stable_baselines3 import PPO
from cacc_ncars_env import CACCNCarsEnv

def main():
    print("Környezet és modell betöltése teszteléshez...")
    
    # Ugyanaz a 3 autó és 36.11 m/s, mint a tanításnál
    env = CACCNCarsEnv(num_cars=4, target_speed=36.11, safe_dist=15.0)
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
        
        for agent, reward in rewards.items():
            total_rewards[agent] += reward
            if reward <= -50.0:
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