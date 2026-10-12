import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import supersuit as ss
from cacc_ncars_env import CACCNCarsEnv

class DirectTensorboardRewardCallback(BaseCallback):
    """
    Közvetlenül a TensorBoard eseményfájlba írja az 1. autó (lead car)
    és a teljes konvoj átlagos lépésenkénti jutalmát.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.writer = None

    def _on_training_start(self) -> None:
        self.writer = SummaryWriter(log_dir=self.logger.dir)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None and len(rewards) > 0 and self.writer is not None:
            lead_car_rew = float(rewards[0])
            mean_step_rew = float(np.mean(rewards))

            self.writer.add_scalar("reward/lead_car_reward", lead_car_rew, self.num_timesteps)
            self.writer.add_scalar("reward/mean_step_reward", mean_step_rew, self.num_timesteps)

            if self.n_calls % 1000 == 0:
                self.writer.flush()
        return True

    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.flush()
            self.writer.close()

def main():
    print("Környezet inicializálása tanításhoz...")
    
    # 130 km/h = 36.11 m/s
    env = CACCNCarsEnv(num_cars=4, target_speed=36.11, safe_dist=72.0)
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        tensorboard_log="./runs/ppo_cacc_training/"
    )

    print("Tanítás indítása (2,000,000 lépés)...")
    model.learn(
        total_timesteps=2e6, 
        tb_log_name="PPO_3cars", 
        callback=DirectTensorboardRewardCallback()
    )

    print("Modell mentése...")
    model.save("ppo_cacc_model")
    print("Tanítás befejezve! Eredmények kimentve a runs/ppo_cacc_training/ mappába.")

if __name__ == "__main__":
    main()