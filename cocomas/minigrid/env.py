from typing import Literal, Any
import functools

import gymnasium as gym
import stable_baselines3 as sb3
import pettingzoo as pz
import numpy as np
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #mps is almost always slower

#from cocomas.path_planning import A_Star, grid, node

class miniEnv(pz.ParallelEnv):
    def __init__(self, 
                 agents: int = 2, 
                 targets: int = 1, 
                 grid_size: int = 20,
                 horizon: int = 10,
                 mode: Literal["static", "dynamic"] = "static", 
                 device: int = DEVICE):
        
        self.n_agents = agents
        self.possible_agents = [i for i in range(self.n_agents)]
        self.possible_agent_names = ["agent{}".format(i) for i in range(self.n_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.n_targets = targets

        self.h = self.w = grid_size
        self.c = 1 # map, agent locations, 

        self.action_dict = {
            0 : [[0,0]], #"no_op",
            1 : [[0,-1]], #"left",
            2 : [[0,1]], #"right",
            3 : [[-1,0]], #"up",
            4 : [[1,0]], #"down",
        }
        self.n_actions = len(self.action_dict.keys())
        self.H = horizon

        self.mode = mode
        self.isDynamic = self.mode == "dynamic"

        self.action_spaces = {agent : gym.spaces.MultiDiscrete([self.n_actions for _ in range(self.H)]) for agent in self.possible_agents}
        self.observation_spaces = {agent : gym.spaces.Dict({
            "map" : gym.spaces.Box(-1, 1, [self.h, self.w, self.c]),
            "prev_plan" : gym.spaces.MultiDiscrete([self.n_actions for _ in range(self.H)]),
            "external_plans" : gym.spaces.Dict({
                ext_agent : gym.spaces.MultiDiscrete([self.n_actions for _ in range(self.H)])
                for ext_agent in range(self.n_agents) if ext_agent != agent
                })
            })
            for agent in self.possible_agents
            }
        
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def close(self):
        pass

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict[Any, dict]]:
        self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.steps = 0

        self.occ_map = np.zeros((self.h, self.w, 1))
        free_idx = np.argwhere(self.occ_map == 0)
        perm_idx = self.rng.permutation(free_idx.shape[0])
        self.agents_loc = free_idx[perm_idx[:self.n_agents]]
        self.targets_loc =free_idx[perm_idx[self.n_agents:self.n_targets]]

        observations = {agent : {
            "map" : self.occ_map,
            "prev_plan" : np.array([0 for _ in range(self.H)]),
            "external_plans" : {
                ext_agent : np.array([0 for _ in range(self.H)])
                for ext_agent in self.agents if ext_agent != agent
                }
            }
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        self.state = observations
        self.prev_actions = {agent : np.array([0 for _ in range(self.H)]) for agent in self.agents}
        return observations, infos
    
    def get_obs(self):
        return {agent : {
            "map" : self.occ_map,
            "prev_plan" : self.prev_actions[agent],
            "external_plans" : {
                ext_agent : self.prev_actions[ext_agent]
                for ext_agent in self.agents if ext_agent != agent
                }
            }
            for agent in self.agents
        }
        
    def get_rew(self):
        return {agent : 0
                for agent in self.agents}


    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        for agent in self.agents:

            self.prev_actions[agent] = np.concatenate((self.prev_actions[agent][1:],actions[agent]))

if __name__ == "__main__":
    env = miniEnv()
    action = {0 : [1], 1: [2]}
    env.reset()
    print(env.prev_actions)
    env.step(action)
    print(env.prev_actions)
        


