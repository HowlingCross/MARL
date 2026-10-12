import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

class CACCNCarsEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "cacc_ncars_v0"}

    def __init__(self, 
                 num_cars=4, 
                 target_speed=36.11, 
                 safe_dist=15.0, 
                 weight_speed=0.5, 
                 weight_dist=0.1, 
                 penalty_crash=-1000.0,
                 max_steps=1000,
                 render_mode=None):
        super().__init__()
        self.render_mode = render_mode     

        self.num_cars = num_cars
        self.possible_agents = [f"car_{i}" for i in range(self.num_cars)]
        self.agents = self.possible_agents[:]
        
        self.dt = 0.1
        self.max_speed = 200.0
        self.max_steps = max_steps
        
        self.target_speed = target_speed
        self.safe_dist = safe_dist
        self.weight_speed = weight_speed
        self.weight_dist = weight_dist
        self.penalty_crash = penalty_crash
        
        self.action_spaces = {}
        self.observation_spaces = {}
        
        for agent in self.possible_agents:
            self.action_spaces[agent] = spaces.Box(low=-5.0, high=3.0, shape=(1,), dtype=np.float32)
            self.observation_spaces[agent] = spaces.Box(
                low=np.array([0.0, 0.0, -self.max_speed], dtype=np.float32),
                high=np.array([self.max_speed, 1000.0, self.max_speed], dtype=np.float32)
            )

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.state = {}
        
        start_gap = self.safe_dist + 5.0 
        for i, agent in enumerate(self.possible_agents):
            self.state[agent] = {
                "pos": float((self.num_cars - i) * start_gap), 
                "vel": 20.0
            }
            
        return self._get_observations(), {agent: {} for agent in self.agents}

    def step(self, actions):
        self.current_step += 1
        
        for agent in self.agents:
            acceleration = np.clip(actions[agent][0], -5.0, 3.0)
            self.state[agent]["vel"] += acceleration * self.dt
            self.state[agent]["vel"] = np.clip(self.state[agent]["vel"], 0.0, self.max_speed)
            self.state[agent]["pos"] += self.state[agent]["vel"] * self.dt

        crash = False
        for i in range(1, self.num_cars):
            agent_front = f"car_{i-1}"
            agent_back = f"car_{i}"
            dist = self.state[agent_front]["pos"] - self.state[agent_back]["pos"]
            if dist <= 0.0:
                crash = True
                break

        observations = self._get_observations()
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        if crash:
            for agent in self.agents:
                rewards[agent] = self.penalty_crash
                terminations[agent] = True
            self.agents = []
            return observations, rewards, terminations, truncations, {a: {} for a in self.possible_agents}
        
        for i, agent in enumerate(self.agents):
            vel = self.state[agent]["vel"]
            speed_rew = -abs(self.target_speed - vel) * self.weight_speed
            
            dist_rew = 0.0
            if i > 0:
                agent_front = f"car_{i-1}"
                dist = self.state[agent_front]["pos"] - self.state[agent]["pos"]
                dist_rew = -abs(self.safe_dist - dist) * self.weight_dist
                
            rewards[agent] = speed_rew + dist_rew
            
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True
            self.agents = []

        return observations, rewards, terminations, truncations, {a: {} for a in self.possible_agents}

    def _get_observations(self):
        obs = {}
        for i, agent in enumerate(self.possible_agents):
            vel = self.state[agent]["vel"]
            if i == 0:
                obs[agent] = np.array([vel, 1000.0, 0.0], dtype=np.float32)
            else:
                agent_front = f"car_{i-1}"
                dist = self.state[agent_front]["pos"] - self.state[agent]["pos"]
                rel_vel = self.state[agent_front]["vel"] - vel
                obs[agent] = np.array([vel, dist, rel_vel], dtype=np.float32)
        return obs

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]