import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
from cacc_ncars_env import CACCNCarsEnv


class ConsoleProgressCallback(BaseCallback):
    def __init__(self, print_freq=15000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.last_print_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_print_step >= self.print_freq:
            self.last_print_step = self.num_timesteps
            if len(self.model.ep_info_buffer) > 0:
                mean_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                mean_len = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                print(f"Lépés: {self.num_timesteps:,} | Átlag reward: {mean_rew:.1f} | Átlag hossz: {mean_len:.0f}/600")
        return True


def make_cacc_env(num_envs=6, num_cpus=6):
    env = CACCNCarsEnv(num_cars=4, target_speed=36.11, safe_dist=22.0)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env


def main():
    log_dir = "./logs/eval_results_ppo/"
    tensorboard_dir = "./tensorboard/"
    model_save_dir = "./models/ppo/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    train_env = make_cacc_env(num_envs=6, num_cpus=6)
    eval_env = make_cacc_env(num_envs=1, num_cpus=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_dir, "best_model"),
        log_path=log_dir,
        eval_freq=4000,          # 6 env * 4 autó * 4000 = 96 000 lépésenként értékel ki
        n_eval_episodes=20,      # 20 tesztpálya átlaga -> sokkal simább eval görbe!
        deterministic=True,
        render=False,
        verbose=1
    )

    callbacks = CallbackList([eval_callback, ConsoleProgressCallback(print_freq=15000)])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        device="cpu",
        verbose=1,
        tensorboard_log=tensorboard_dir
    )

    try:
        model.learn(
            total_timesteps=20000000,
            tb_log_name="PPO_CACC",
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTanítás megszakítva, jelenlegi modell mentése...")

    save_path = os.path.join(model_save_dir, "ppo_cacc_model")
    model.save(save_path)
    print(f"Modell mentve ide: {save_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()