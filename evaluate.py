import numpy as np
from stable_baselines3 import PPO, SAC
from cacc_ncars_env import CACCNCarsEnv


def main():
    print("Környezet és modell betöltése vizualizációval...")

    env = CACCNCarsEnv(num_cars=2, target_speed=36.0, safe_dist=25.0, render_mode="human")
    model = PPO.load("./models/ppo/best_model/best_model")
    obs, infos = env.reset()

    total_rewards = {agent: 0.0 for agent in env.possible_agents}
    steps = 0
    failed = False

    print("Szimuláció futtatása...")
    while env.agents:
        actions = {}
        for agent in env.agents:
            action, _states = model.predict(obs[agent], deterministic=True)
            actions[agent] = action

        obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent, reward in rewards.items():
            total_rewards[agent] += reward
            if terminations[agent]:
                failed = True

        steps += 1

    print("\n--- TESZT EREDMÉNYEK ---")
    print(f"Lefutott időlépések: {steps} / {env.max_steps}")
    print(f"Ütközés vagy leszakadás történt: {'IGEN' if failed else 'NEM'}")
    print("\nÖsszjutalom és végső állapot autónként:")
    for i, agent in enumerate(env.possible_agents):
        front_pos = env.lead_pos if i == 0 else env.state[f"car_{i-1}"]["pos"]
        dist = front_pos - env.state[agent]["pos"]
        vel = env.state[agent]["vel"]
        print(f"  {agent}: Jutalom = {total_rewards[agent]:.2f} | Távolság = {dist:.2f} m | Sebesség = {vel:.2f} m/s")

    env.close()

if __name__ == "__main__":
    main()