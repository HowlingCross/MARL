import os
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from cacc_ncars_env import CACCNCarsEnv

# Válassz modellt: "ppo", "sac", "td3", vagy "classic"
ALGO = "td3"
NUM_EPISODES = 2
RENDER = True


def classic_cacc_policy(env, agent):
    """
    Klasszikus szabályalapú CACC (IDM hosszirányú követés + résalapú besorolás).
    """
    st = env.state[agent]
    pos = st["pos"]
    vel = st["vel"]
    lane = st["lane"]

    # 1. Hosszirányú gyorsulás (IDM + virtuális jármű vetítés a felhajtóról)
    front_pos, front_vel, _ = env._get_leader_in_lane(
        pos, lane, exclude_agent=agent, include_virtual=True
    )
    s = max(0.5, front_pos - pos)
    dv = vel - front_vel  # közeledési sebesség

    a_max = 2.5
    b_comf = 3.0
    v0 = env.target_speed
    s0 = 8.0
    T = 0.45  # CACC időköz (s)

    s_star = s0 + max(0.0, vel * T + (vel * dv) / (2.0 * np.sqrt(a_max * b_comf)))
    acc = a_max * (1.0 - (vel / v0) ** 4 - (s_star / s) ** 2)
    acc = float(np.clip(acc, -6.0, 4.0))

    # 2. Sávváltási döntés (MOBIL-jellegű biztonsági résellenőrzés)
    lane_cmd = 1.0
    if lane == 1 and pos >= env.merge_zone_start:
        front_gap, _, rear_gap, _ = env._get_mainline_gaps(pos, exclude_agent=agent)
        if front_gap > 11.0 and rear_gap > 11.0:
            lane_cmd = -1.0

    return np.array([acc, lane_cmd], dtype=np.float32)


def load_model(algo):
    if algo == "classic":
        return None
    elif algo == "ppo":
        return PPO.load("./models/ppo/ppo_cacc_model", device="cpu")
    elif algo == "sac":
        best = "./models/sac/best_model/best_model.zip"
        path = best if os.path.exists(best) else "./models/sac/sac_cacc_final"
        return SAC.load(path, device="cpu")
    elif algo == "td3":
        best = "./models/td3/best_model/best_model.zip"
        path = best if os.path.exists(best) else "./models/td3/td3_cacc_final"
        return TD3.load(path, device="cpu")
    else:
        raise ValueError(f"Ismeretlen algoritmus: {algo}")


def main():
    print(f"[INFO] {ALGO.upper()} vezérlő és környezet betöltése...")
    model = load_model(ALGO)
    env = CACCNCarsEnv(
        num_cars=4,
        target_speed=36.11,
        safe_dist=22.0,
        render_mode="human" if RENDER else None
    )

    success_count = 0
    all_rewards = []
    all_jerks = []

    for ep in range(1, NUM_EPISODES + 1):
        obs, infos = env.reset(seed=100 + ep)  # Fix seedek, hogy ugyanazokon a pályákon mérj!
        ep_rewards = {a: 0.0 for a in env.possible_agents}
        prev_acc = {a: 0.0 for a in env.possible_agents}
        ep_jerks = []
        steps = 0

        while env.agents:
            actions = {}
            for agent in env.agents:
                if ALGO == "classic":
                    act = classic_cacc_policy(env, agent)
                else:
                    act, _ = model.predict(obs[agent], deterministic=True)
                actions[agent] = act
                acc_val = float(np.clip(act[0], -6.0, 4.0))
                ep_jerks.append(abs(acc_val - prev_acc[agent]) / env.dt)
                prev_acc[agent] = acc_val

            obs, rewards, terminations, truncations, infos = env.step(actions)
            for a, r in rewards.items():
                ep_rewards[a] += r
            steps += 1

        crashed = env.crash_happened or env.ramp_overrun
        merged_ok = env.merge_completed and not crashed
        if merged_ok:
            success_count += 1

        mean_ep_rew = float(np.mean(list(ep_rewards.values())))
        mean_ep_jerk = float(np.mean(ep_jerks)) if ep_jerks else 0.0
        all_rewards.append(mean_ep_rew)
        all_jerks.append(mean_ep_jerk)

        print(f"Ep {ep:02d} | Lépés: {steps:3d}/600 | Ütközés: {'IGEN' if crashed else 'NEM '} | Besorolás: {'OK ' if merged_ok else 'NEM'} | Jutalom: {mean_ep_rew:6.1f} | Jerk: {mean_ep_jerk:4.2f} m/s³")

    print("\n==========================================")
    print(f"ÖSSZESÍTÉS ({ALGO.upper()} - {NUM_EPISODES} azonos tesztpálya):")
    print(f"Sikeres, ütközésmentes besorolás: {success_count}/{NUM_EPISODES} ({success_count/NUM_EPISODES*100:.0f}%)")
    print(f"Átlagos összjutalom: {np.mean(all_rewards):.2f}")
    print(f"Átlagos rántás (jerk): {np.mean(all_jerks):.2f} m/s³")
    print("==========================================")
    env.close()


if __name__ == "__main__":
    main()