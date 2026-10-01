from stable_baselines3 import PPO
import supersuit as ss
from cacc_ncars_env import CACCNCarsEnv

def main():
    print("Környezet inicializálása tanításhoz...")
    env = CACCNCarsEnv(num_cars=3, target_speed=25.0, safe_dist=15.0)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        tensorboard_log="./runs/ppo_cacc_training/"
    )

    print("Tanítás indítása (100,000 lépés)...")
    model.learn(total_timesteps=100000, tb_log_name="PPO_3cars")
    model.save("ppo_cacc_model")
    print("Tanítás befejezve! Eredmények kimentve a runs/ppo_cacc_training/ mappába.")

if __name__ == "__main__":
    main()
