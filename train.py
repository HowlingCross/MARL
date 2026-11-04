import os
import supersuit as ss
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from cacc_ncars_env import CACCNCarsEnv


def make_cacc_env():
    """
    Létrehozza a többágenses környezetet és SuperSuit segítségével 
    vektorizált, egyágenses felületűvé (Parameter Sharing) alakítja.
    """
    env = CACCNCarsEnv()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env


def main():
    log_dir = "./logs/eval_results/"
    tensorboard_dir = "./tensorboard/"
    model_save_dir = "./models/ppo"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    print("[INFO] Környezetek inicializálása...")
    train_env = make_cacc_env()
    eval_env = make_cacc_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_dir, "best_model"),
        log_path=log_dir,
        eval_freq=5000,          
        n_eval_episodes=5,       
        deterministic=True,      
        render=False,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=model_save_dir,
        name_prefix="cacc_model_step",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    
    print("[INFO] PPO modell felépítése...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        verbose=1,
        tensorboard_log=tensorboard_dir
    )
    final_model_path = os.path.join(model_save_dir, "ppo_cacc_final")

    total_timesteps = 6000000
    print(f"[INFO] Tanítás kezdete ({total_timesteps} időlépés)...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    model.save(final_model_path)
    print(f"[INFO] Tanítás befejezve! Modell mentve ide: {final_model_path}")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()