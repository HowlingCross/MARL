import os
import supersuit as ss
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from cacc_ncars_env import CACCNCarsEnv


def make_cacc_env(num_envs=6, num_cpus=6):
    env = CACCNCarsEnv(num_cars=4, target_speed=36.11, safe_dist=22.0)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env


def main():
    total_timesteps = 13000000
    log_dir = "./logs/eval_results_sac/"
    tensorboard_dir = "./tensorboard/"
    model_save_dir = "./models/sac/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    train_env = make_cacc_env(num_envs=8, num_cpus=8)
    eval_env = make_cacc_env(num_envs=1, num_cpus=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_dir, "best_model"),
        log_path=log_dir,
        eval_freq=2000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=15000,
        save_path=model_save_dir,
        name_prefix="cacc_sac_step",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    print("[INFO] SAC modell felépítése...")
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        train_freq=(6, "step"),
        gradient_steps=4,
        gamma=0.99,
        tau=0.005,
        device="cuda",
        verbose=1,
        tensorboard_log=tensorboard_dir
    )
    final_model_path = os.path.join(model_save_dir, "sac_cacc_final")

    print(f"[INFO] Tanítás kezdete ({total_timesteps} időlépés)...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="SAC_CACC",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[INFO] Tanítás megszakítva, jelenlegi modell mentése...")

    model.save(final_model_path)
    print(f"[INFO] Tanítás befejezve! Modell mentve ide: {final_model_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()