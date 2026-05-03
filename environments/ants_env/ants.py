import time
import random
import sys
import numpy as np
from typing import Tuple, Dict

from scipy.ndimage import gaussian_filter
from gymnasium.spaces import Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType


# AWAY, MID, NEAR
N_POS   = ((310, 30), (310, 50), (310, 70), (310, 90), (310, 110))
NE_POS  = ((590, 30), (570, 50), (550, 70), (530, 90), (510, 110))    
E_POS   = ((590, 310), (570, 310), (550, 310), (530, 310), (510, 310))   
SE_POS  = ((590, 590), (570, 570), (550, 550), (530, 530), (510, 510))  
S_POS   = ((310, 590), (310, 570), (310, 550), (310, 530), (310, 510)) 
SW_POS  = ((30, 590), (50, 570), (70, 550), (90, 530), (110, 510)) 
W_POS   = ((30, 310), (50, 310), (70, 310), (90, 310), (110, 310))
NW_POS  = ((30, 30), (50, 50), (70, 70), (90, 90), (110, 110)) 
FOOD_POS = (N_POS, NE_POS, E_POS, SE_POS, S_POS, SW_POS, W_POS, NW_POS)


class Ants(AECEnv):

    def observe(self, agent: str) -> ObsType:
        return np.array(self.observations[agent]) 

    def __init__(self, seed: int, eval: bool = False, fixed_food: bool = False, **kwargs: Dict) -> None:
        np.random.seed(seed)
        random.seed(seed)

        self.eval = eval
        self.fixed_food = fixed_food
        
        self.num_learners = kwargs['learners'] 
        self.sniff_threshold = kwargs['sniff_threshold']
        self.diffuse_area = kwargs['diffuse_area']
        self.diffuse_radius = kwargs['diffuse_radius']
        self.lay_area = kwargs['lay_area']
        self.lay_amount_food = kwargs['lay_amount']
        self.lay_amount = np.array([self.lay_amount_food for _ in range(self.num_learners)])
        self.lay_amount_min = kwargs['lay_amount_min']
        self.lay_amount_first = kwargs['lay_amount_first']
        self.first_drop = np.array([True for _ in range(self.num_learners)])
        
        self.ph_decay = kwargs['ph_decay']
        self.evaporation = kwargs['evaporation']
        self.MAX_TICKS = kwargs['max_episode_ticks']
    
        self.food_quantity = kwargs['food_quantity']

        self.reward_type = kwargs['reward_type']
        assert(
            self.reward_type in ("ind", "glob", "mix")
        ), "Error! reward_type admitted values are:\n'ind' - only individual reward;\n'glob' - only global reward;\n'mix' - combination of both rewards."
        self.penalty = kwargs['penalty']
        self.nest_penalty = kwargs['nest_penalty']
        self.ind_rew_scale_1 = kwargs["ind_rew_scale_1"]
        self.ind_rew_scale_2 = kwargs["ind_rew_scale_2"]
        self.glob_rew_scale_1 = kwargs["glob_rew_scale_1"]
        self.glob_rew_scale_2 = kwargs["glob_rew_scale_2"]

        self.W = kwargs['W']
        self.H = kwargs['H']
        assert (self.W == self.H), "Error! W and H must be equal."
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']

        self.N_DIRS = 8
        self.sniff_patches = kwargs['sniff_patches']
        assert (
            self.sniff_patches in (3, 5)
        ), "Error! sniff_patches admitted values are 3 or 5."
        self.wiggle_patches = kwargs['wiggle_patches'] 
        assert (
            self.wiggle_patches in (3, 5)
        ), "Error! wiggle_patches admitted values are 3 or 5"
        # Used to calculate the agent's directions.
        # It's a personal convention.

        self.coords = []
        self.offset = self.patch_size // 2
        self.W_pixels = self.W * self.patch_size
        self.H_pixels = self.H * self.patch_size
        
        for x in range(self.offset, (self.W_pixels - self.offset) + 1, self.patch_size):
            for y in range(self.offset, (self.H_pixels - self.offset) + 1, self.patch_size):
                self.coords.append((x, y))  # "centre" of the patch or turtle (also ID of the patch)
        
        self.pos_to_idx = {}
        ids = [(i, j) for i in range(self.W) for j in range(self.H)]
        for p, i in zip(self.coords, ids):
            self.pos_to_idx[p] = i
        
        self.possible_agents = [str(i) for i in range(self.num_learners)]  # DOC learning agents IDs
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent = self._agent_selector.reset()

        n_coords = len(self.coords)
        self.MIN_COORD = min(self.coords)
        self.MAX_COORD = max(self.coords)
        # patches-own [chemical] - amount of pheromone in each patch
        self.patches = {
            self.coords[i]: {
                'id': i,
                'chemical_0': 0.0,
                'chemical_1': 0.0,
                'turtles': []
            }
            for i in range(n_coords)
        }

        # pre-compute relevant structures to speed-up computation during rendering steps
        self.lay_patches = self._find_neighbours(self.lay_area)
        
        # Agent's field of view
        self.fov, self.fov_dirs = self._field_of_view(self.wiggle_patches)
        # Agent's pheromone field of view
        self.ph_fov, self.ph_fov_dirs = self._field_of_view(self.sniff_patches)

        self.actions = kwargs['actions']
        self._action_spaces = {
            a: Discrete(len(self.actions))
            for a in self.possible_agents
        }  
        self.POSSIBLE_ACTIONS = 4
        
        self._observation_spaces = {
            a: Box(low=0.0, high=np.inf, shape=(self.sniff_patches * 2 + 2,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.POSSIBLE_STATES = 2**3
    
        self.FINITE_FOOD = True if self.eval else False
        self.FOOD_SOURCES = 3

        if self.fixed_food:
            self.NEST_POS = (130, 510)
        else:
            self.NEST_POS = (310, 310)
        
        self.count_pos = np.zeros((self.W, self.H), dtype=np.uint64)
        
        # create learners turtle
        self.learners = {
            i: {
                'pos': self.NEST_POS,
                'dir': np.random.randint(self.N_DIRS), 
                'flags': [0, 0, 0] 
            }
            for i in range(self.num_learners)
        }
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtles

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.num_learners)))
        )
    
    def _field_of_view(self, patches: int) -> Tuple[Dict, Dict]:
        """
        Pre-compute every possible agent's direction
        """

        movements = np.array([
            (0, -self.patch_size),                  # dir 0
            (self.patch_size, -self.patch_size),    # dir 1
            (self.patch_size, 0),                   # dir 2
            (self.patch_size, self.patch_size),     # dir 3
            (0, self.patch_size),                   # dir 4
            (-self.patch_size, self.patch_size),    # dir 5
            (-self.patch_size, 0),                  # dir 6
            (-self.patch_size, -self.patch_size),   # dir 7
        ])
        fov = {}
        fov_dirs = {}
        
        if patches < self.N_DIRS:
            central = patches // 2
            sliding_window = []
            
            for i in range(self.N_DIRS):
                tmp = []
                for j in range(patches):
                    tmp.append((i + j) % self.N_DIRS)
                sliding_window.append(tmp)
            sliding_window = sorted(sliding_window, key=lambda x: x[central])

            for c in self.coords:
                tmp_fov = movements + c
                
                v = tmp_fov[sliding_window,:] 
                fov[c] = v
                
                # Rule: if the agent is facing perpendicular to the obstacle, it reverses direction; 
                # otherwise, its direction is approximated by the nearest position.
                x = np.any((v < self.MIN_COORD[0]) | (v > self.MAX_COORD[0]), axis=-1)
                y = np.all(x, axis=-1)
                z = np.where(y == True)[0]
                if z.size != 0:
                    w = (z + 4) % self.N_DIRS   # 4 is the offset to the opposite direction
                    s = np.stack((z, w))
                    fov_dirs[c] = s
        else:
            for c in self.coords:
                v = movements + c
                x = np.any((v < self.MIN_COORD[0]) | (v > self.MAX_COORD[0]), axis=-1)
                z = np.where(x == True)[0]
                e = (z + 4) % self.N_DIRS   # 4 is the offset to the opposite direction
                v[z] = v[e]
                tmp_fov = np.clip(v, self.MIN_COORD[0], self.MAX_COORD[0])  # approximate to the nearest pos
                fov[c] = tmp_fov

        return fov, fov_dirs
    
    def _find_neighbours(self, area: int) -> Dict:
        """
        For each patch, find neighbouring patches within square radius 'area'
        """

        neighbours = {}
        min_x = self.MIN_COORD[0]
        min_y = self.MIN_COORD[1]
        max_x = self.MAX_COORD[0]
        max_y = self.MAX_COORD[1]

        for p in self.patches:
            neighbours[p] = []
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    if (x >= min_x and y >= min_y) and (x <= max_x and y <= max_y):
                        neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    if (x >= min_x and y >= min_y) and (x <= max_x and y <= max_y):
                        neighbours[p].append((x, y))
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    if (x >= min_x and y >= min_y) and (x <= max_x and y <= max_y):
                        neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    if (x >= min_x and y >= min_y) and (x <= max_x and y <= max_y):
                        neighbours[p].append((x, y))
            neighbours[p] = list(set(neighbours[p]))

        return neighbours

    def _get_new_positions(
        self,
        possible_patches: Dict,
        agent: Dict,
        random_walk: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Compute agent's new position.
        """

        pos = agent["pos"]
        dir = agent["dir"]
        fov = self.fov_dirs if random_walk else self.ph_fov_dirs

        if pos in fov.keys():
            # Convenzione del momento:
            # se l'agente ha una direzione perpendicolare all'ostacolo inverte la sua direzione,
            # altrimenti viene approssimata con la posizione più vicina.
            i, j = np.where(fov[pos] == dir)
            if i.size != 0 and j.size != 0:
                new_dir = fov[pos][-1][j].item()
            else:
                new_dir = dir
            new_pos = possible_patches[pos][new_dir]
        else:
            if len(possible_patches[pos].shape) > 2:
                new_pos = possible_patches[pos][dir]
            else:
                new_pos = possible_patches[pos]    
            new_dir = dir

        return new_pos, new_dir
    
    def _get_idx_dir(self, pos: np.ndarray) -> int:
        x = np.any((pos < self.MIN_COORD[0]) | (pos > self.MAX_COORD[0]), axis=-1)
        y = np.where(x == False)[0]
        idx_dir = np.random.choice(y)

        return idx_dir.item()
    
    def _get_new_direction(self, n_patches: int, old_dir: int, idx_dir: int) -> int:
        """
        Compute agent's new direction.
        """

        start = (old_dir - (n_patches // 2)) % self.N_DIRS 
        new_dirs = np.array([(i + start) % self.N_DIRS for i in range(n_patches)])

        return new_dirs[idx_dir].item()

    def _random_walk(self, patches: Dict, turtle: Dict) -> Tuple[Dict, Dict]:
        f, direction = self._get_new_positions(self.fov, turtle, True)
        patches[turtle['pos']]['turtles'].remove(self.agent)
        idx_dir = self._get_idx_dir(f)
        turtle["pos"] = tuple(f[idx_dir])
        patches[turtle['pos']]['turtles'].append(self.agent)

        if self.wiggle_patches < self.N_DIRS:
            turtle["dir"] = self._get_new_direction(self.wiggle_patches, direction, idx_dir)
        else:
            turtle["dir"] = idx_dir
        
        turtle = self._check_pos(turtle)

        return patches, turtle

    def random_walk(self) -> None:
        """
        Action 0: move in random direction.
        """      

        self.patches, self.learners[self.agent] = self._random_walk(self.patches, self.learners[self.agent])

    def _find_max_pheromone(self, agent: Dict, obs: np.ndarray) -> Tuple[float, Tuple, int]:
        # Det = follow greatest pheromone
        f, direction = self._get_new_positions(self.ph_fov, agent)
        x = np.any((f < self.MIN_COORD[0]) | (f > self.MAX_COORD[0]), axis=-1)
        y = np.where(x == False)[0]
        idx = y[obs[y].argmax()]
        ph_pos = tuple(f[idx])
        ph_val = obs[idx]
        
        if self.sniff_patches < self.N_DIRS:
            ph_dir = self._get_new_direction(self.sniff_patches, direction, idx)
        else:
            ph_dir = idx

        return ph_val.item(), ph_pos, ph_dir

    def _follow_pheromone(self, patches: Dict, ph_coords: Tuple, ph_dir: int, turtle: Dict) -> Tuple[Dict, Dict]:
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = ph_coords
        patches[turtle['pos']]['turtles'].append(self.agent)
        turtle["dir"] = ph_dir
        
        turtle = self._check_pos(turtle)
        
        return patches, turtle

    def _lay_pheromone(self, patches: Dict, pos: Tuple) -> Dict:
        """
        Drop agent's pheromone 
        """

        if self.first_drop[self.agent] and (self.learners[self.agent]['flags'] == [1, 0, 0] or self.learners[self.agent]['flags'] == [1, 0, 1]):
            ph = self.lay_amount_food * self.lay_amount_first
            self.first_drop[self.agent] = False 
        elif self.learners[self.agent]['flags'] == [1, 0, 0] or self.learners[self.agent]['flags'] == [1, 0, 1]:
            ph = max(self.lay_amount[self.agent] * self.ph_decay, self.lay_amount_min)
        else:
            ph = self.lay_amount_min 

        self.lay_amount[self.agent] = ph 
        
        for p in self.lay_patches[pos]:
            patches[p]['chemical_0'] +=  ph 

        return patches
                
    def follow_ph_0(self) -> None:
        """
        Action 1: follow agent's pheromone. 
        """

        max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone(
            self.learners[self.agent],
            self.observations[str(self.agent)][:self.sniff_patches]        
        )
        if max_pheromone >= self.sniff_threshold:
            self.patches, self.learners[self.agent] = self._follow_pheromone(
                self.patches,
                max_coords,
                max_ph_dir,
                self.learners[self.agent]
            )
    
    def follow_ph_1(self) -> None:
        """
        Action 2: follow the nest's pheromone. 
        """

        max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone(
            self.learners[self.agent],
            self.observations[str(self.agent)][self.sniff_patches:-2]        
        )
        if max_pheromone >= self.sniff_threshold:
            self.patches, self.learners[self.agent] = self._follow_pheromone(
                self.patches,
                max_coords,
                max_ph_dir,
                self.learners[self.agent]
            )
    
    def drop_ph_0_and_follow_ph_1(self) -> None:
        """
        Action 3: Drop agent's pheromone while following the nest's pheromone.
        """
        
        max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone(
            self.learners[self.agent],
            self.observations[str(self.agent)][self.sniff_patches:-2]        
        )
        if max_pheromone >= self.sniff_threshold:
            self.patches = self._lay_pheromone(self.patches, self.learners[self.agent]['pos'])
            self.patches, self.learners[self.agent] = self._follow_pheromone(
                self.patches,
                max_coords,
                max_ph_dir,
                self.learners[self.agent]
            )

    def _reset_flags(self, agent: int) -> None:
        self.learners[agent]['flags'] = [0, 0, 0]
    
    def _count_food(self, agent: int) -> None:
        self.food_x_agent[agent] += 1
    
    def _count_food_nest(self) -> None:
        self.food_nest += 1
    
    #def _try_reward_1(self, rewards):
    #    return np.array(rewards).mean()
    
    #def _try_reward_2(self, rewards):
    #    return min(rewards) 
    
    #def _try_reward_3(self, rewards):
    #    return np.array(rewards).sum()
    
    #def _try_reward_4(self, agent):
    #    return self.food_x_agent[agent]
    
    #def _tyr_reward_5(self):
    #    return (self.food_x_agent.mean() / self.current_ticks) 

    def _get_ind_reward(self, agent: int) -> float:
        """
        Get individual reward.
        """

        return (self.food_x_agent[agent] / self.current_ticks).item() * self.ind_rew_scale_1
    
    def _get_global_reward(self) -> float:
        """
        Get global reward.
        """

        return (self.food_nest / self.current_ticks) * self.glob_rew_scale_1 
    
    def _get_reward(self) -> float:
        """
        Get one of these rewards:
            - ind: individual reward
            - glob: global reward
            - mix: individual + global
            - penalty
            - nest_penalty
        """

        if self.learners[self.agent]['pos'] in self.patches_food_1 and self.learners[self.agent]['flags'] == [1, 0, 0]:
            self.learners[self.agent]['flags'][2] = 1

            self._count_food(self.agent)
            
            if self.FINITE_FOOD:
                self.food_counts[self.learners[self.agent]['pos']] -= 1
                self.time_food_1.append(self.current_ticks)
            
            if self.food_counts[self.learners[self.agent]['pos']] == 0:
                self.patches_food_1.remove(self.learners[self.agent]['pos'])
            
            if self.reward_type == "ind":
                return self._get_ind_reward(self.agent)
            elif self.reward_type == "glob":
                return self.penalty
            elif self.reward_type == "mix":
                return self._get_ind_reward(self.agent)
            else:
                raise ValueError(f"Error: reward type {type} is unknown!")
        elif self.learners[self.agent]['pos'] in self.patches_food_2 and self.learners[self.agent]['flags'] == [1, 0, 0]:
            self.learners[self.agent]['flags'][2] = 1

            self._count_food(self.agent)

            if self.FINITE_FOOD:
                self.food_counts[self.learners[self.agent]['pos']] -= 1
                self.time_food_2.append(self.current_ticks)
            
            if self.food_counts[self.learners[self.agent]['pos']] == 0:
                self.patches_food_2.remove(self.learners[self.agent]['pos'])
            
            if self.reward_type == "ind":
                return self._get_ind_reward(self.agent)
            elif self.reward_type == "glob":
                return self.penalty
            elif self.reward_type == "mix":
                return self._get_ind_reward(self.agent)
            else:
                raise ValueError(f"Error: reward type {type} is unknown!")
        elif self.learners[self.agent]['pos'] in self.patches_food_3 and self.learners[self.agent]['flags'] == [1, 0, 0]:
            self.learners[self.agent]['flags'][2] = 1

            self._count_food(self.agent)
            
            if self.FINITE_FOOD:
                self.food_counts[self.learners[self.agent]['pos']] -= 1
                self.time_food_3.append(self.current_ticks)
            
            if self.food_counts[self.learners[self.agent]['pos']] == 0:
                self.patches_food_3.remove(self.learners[self.agent]['pos'])

            if self.reward_type == "ind":
                return self._get_ind_reward(self.agent)
            elif self.reward_type == "glob":
                return self.penalty
            elif self.reward_type == "mix":
                return self._get_ind_reward(self.agent)
            else:
                raise ValueError(f"Error: reward type {type} is unknown!")
        elif self.learners[self.agent]['pos'] in self.patches_nest and self.learners[self.agent]['flags'] == [1, 1, 1]:
            self._count_food_nest()
            
            self._reset_flags(self.agent)
            
            self.first_drop[self.agent] = True
            
            if self.reward_type == "ind":
                return self._get_ind_reward(self.agent) * self.ind_rew_scale_2
            elif self.reward_type == "glob":
                return self._get_global_reward() * self.glob_rew_scale_2
            elif self.reward_type == "mix":
                return self._get_global_reward()
            else:
                raise ValueError(f"Error: reward type {type} is unknown!")
        elif self.learners[self.agent]['pos'] in self.patches_nest and self.learners[self.agent]['flags'] == [0, 0, 0]:
            return self.nest_penalty
        else:
            return self.penalty
    
    def _get_obs(self, agent: Dict) -> np.ndarray:
        """
        Get agent's observations
        """
        f, _ = self._get_new_positions(self.ph_fov, agent)

        x = np.any((f < self.MIN_COORD[0]) | (f > self.MAX_COORD[0]), axis=-1)
        y = np.where(x == False)[0]
        new_f = f[y]

        obs_ph_0 = np.zeros(self.sniff_patches)
        obs_ph_0[y] += np.array([self.patches[tuple(i)]["chemical_0"] for i in new_f])

        obs_ph_1 = np.zeros(self.sniff_patches)
        obs_ph_1[y] += np.array([self.patches[tuple(i)]["chemical_1"] for i in new_f])

        flags = np.zeros(2)
        obs = np.concatenate((obs_ph_0, obs_ph_1, flags))
        
        if agent['flags'] != [0, 0, 0]:
            obs[-2] = 1

        if agent['pos'] in self.patches_nest:
            obs[-1] = 1 

        return obs
    
    def _process_agent(self, rewards_cust: Dict) -> Tuple[np.ndarray, Dict]:
        """
        In this methods we compute the agent's reward and it's observation.
        """

        observations = self._get_obs(self.learners[self.agent])
        reward = self._get_reward()
        rewards_cust[self.agent].append(reward)

        return observations, rewards_cust

    def _check_pos(self, turtle: Dict) -> Dict:
        """
        Check whether the agent has found food or brought it back to the nest.
        """

        if turtle['pos'] in self.patches_food_1 and turtle['flags'] == [0, 0, 0]: 
            turtle['flags'][0] = 1
        elif turtle['pos'] in self.patches_food_2 and turtle['flags'] == [0, 0, 0]: 
            turtle['flags'][0] = 1
        elif turtle['pos'] in self.patches_food_3 and turtle['flags'] == [0, 0, 0]: 
            turtle['flags'][0] = 1
        elif turtle['pos'] in self.patches_nest and turtle['flags'] == [1, 0, 1]: 
            turtle['flags'][1] = 1

        return turtle

    def _save_rewards(self) -> None:
        for ag in self.agents:
            reward = self.rewards_cust[self.agent_name_mapping[ag]][-1]
            self.rewards[ag] = round(reward, 2)

    def _diffuse_and_evaporate(self, patches: Dict) -> Dict:
        """
        This diffuse method use a gaussian filter for the process.
        This is a kind of parallel diffusion.
        Evaporates pheromone from each patch according to param self.evaporation
        """

        # Diffusion
        grid = np.array([patches[p]["chemical_0"] for p in patches.keys()]).reshape((self.W, self.H))
        
        if self.diffuse_radius == 0:
            grid = gaussian_filter(grid, sigma=self.diffuse_area, mode='constant')
        else:
            grid = gaussian_filter(grid, sigma=self.diffuse_area, radius=self.diffuse_radius, mode='constant')
        
        grid = grid.flatten()
        # Evaporation
        grid *= self.evaporation
        # Write values
        for p, g in zip(patches, grid):
            patches[p]['chemical_0'] = g
        
        return patches
    
    def _check_food(self) -> None:
        """
        Check to see if all the food has been eaten.
        """
        
        if(
            len(self.patches_food_1) == 0 and
            len(self.patches_food_2) == 0 and
            len(self.patches_food_3) == 0
        ):
            self.no_food = True

    def _check_termination(self) -> None:
        """
        Check run termination. This could happen if there's no more food or a tick threshold has been exceeded. 
        """

        self._check_food()

        tmp = []
        for agent in self.learners: 
            if self.learners[agent]['flags'] == [0, 0, 0]:
                tmp.append(True)
            else:
                tmp.append(False)
        
        if (all(tmp) and self.no_food) or self.current_ticks == self.MAX_TICKS:
            self.done = True

    def step(self, action: int) -> None:
        if(self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        self.agent = self.agent_name_mapping[self.agent_selection]  # ID of agent

        if action == 0:     # Random walk
            self.random_walk()
        elif action == 1:   # Follow pheromone 0
            self.follow_ph_0()
        elif action == 2:   # Follow pheromone 1
            self.follow_ph_1()
        elif action == 3:   # Drop pheromone 0 Follow pheromone 1 
            self.drop_ph_0_and_follow_ph_1()
        else:
            raise ValueError("Action out of range!")

        self.observations[str(self.agent)], self.rewards_cust = self._process_agent(self.rewards_cust)

        if self._agent_selector.is_last():
            self._save_rewards()
            self.patches = self._diffuse_and_evaporate(self.patches)
            self.current_ticks += 1
        else:
            self._clear_rewards()
         
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[str(self.agent)] = 0
        self._accumulate_rewards()
        self._check_termination()
        
    def _sample_food_pos(self) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Sample new foods' positions. 
        """
        
        if self.eval:
            if self.fixed_food: 
                return (510, 50), (570, 290), (70, 230) 
            else:
                pos = random.sample(FOOD_POS, self.FOOD_SOURCES)
                return random.sample(pos[0], 1)[0], random.sample(pos[1], 1)[0], random.sample(pos[2], 1)[0]
        else:
            pos= ((550, 70), (550, 550), (70, 550), (70, 70)) # NE - SE - SO - NO
            return random.sample(pos, self.FOOD_SOURCES)

    def _lay_nest_pheromone(self, patches: Dict) -> Dict:
        """
        Lay nest's pheromone
        """
        grid = np.zeros(self.W * self.H)

        for p in self.patches_nest:
            nest_id = patches[p]['id']
            grid[nest_id] = 1500 if self.fixed_food else 450

        if self.fixed_food:
            grid = gaussian_filter(grid.reshape((self.W, self.H)), sigma=18, radius=None, mode='constant').flatten()
        else:
            grid = gaussian_filter(grid.reshape((self.W, self.H)), sigma=12, radius=None, mode='constant').flatten()
        
        for p, g in zip(patches, grid):
            patches[p]['chemical_1'] = g
        
        return patches
            
    def reset(self, seed=None, return_info=True, options=None) -> None:
        """
        Reset env.
        """
        # empty stuff
        #Different from AECEnv attribute self.rewards - only keeps last step rewards
        self.rewards_cust = {i: [] for i in range(self.num_learners)}
        self.current_ticks = 1
        
        #Initialize attributes for PettingZoo Env
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        
        # re-position learner turtle
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].remove(l)
            self.learners[l]['pos'] = self.NEST_POS
            self._reset_flags(l)
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtle

        self.food_pos_1, self.food_pos_2, self.food_pos_3 = self._sample_food_pos()
        tmp = self._find_neighbours(1)
        self.patches_food_1 = tmp[self.food_pos_1]
        tmp = self._find_neighbours(1)
        self.patches_food_2 = tmp[self.food_pos_2]
        tmp = self._find_neighbours(1)
        self.patches_food_3 = tmp[self.food_pos_3]
        tmp = self._find_neighbours(1)
        self.patches_nest = tmp[self.NEST_POS]
        food_pos = []
        food_pos.extend(self.patches_food_1)
        food_pos.extend(self.patches_food_2)
        food_pos.extend(self.patches_food_3)
        self.food_counts = {p: self.food_quantity for p in food_pos}
        
        self.no_food = False
        self.done = False

        self.time_food_1 = []
        self.time_food_2 = []
        self.time_food_3 = []

        self.food_x_agent = np.zeros(self.num_learners, dtype=np.uint16)
        self.food_nest = 0

        # patches-own [chemical] - amount of pheromone in the patch
        for p in self.patches:
            self.patches[p]['chemical_0'] = 0.0
        self.patches = self._lay_nest_pheromone(self.patches)

        self.observations = {
            a: np.zeros(self.sniff_patches * 2 + 2, dtype=np.float32)
            for a in self.agents
        }
        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
    
    def convert_observation(self, obs: np.ndarray) -> int:
        """
        This method returns the conversion of the observation to an integer.
        It's useful for IQL.
        """

        obs_ph_0 = obs[:self.sniff_patches]
        food = int(obs[-2])
        nest = int(obs[-1])
        
        if np.any(obs_ph_0 >= self.sniff_threshold):
            obs_ph_0_id = 1
        else:
            obs_ph_0_id = 0
        
        obs_id = (obs_ph_0_id * 2**2) + (food * 2**1) + (nest * 2**0)

        return obs_id
    
    def mask_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Mask agent's actions.
        """

        actions_num = len(self.actions)
        actions = [i for i in range(actions_num)]
        food = bool(int(obs[-2]))

        if not food:
            actions.remove(3)              

        if np.all(obs[:self.sniff_patches] < self.sniff_threshold):
            actions.remove(1)
        
        return np.array(actions, dtype=np.uint8)
    

import pygame

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
SKY_BLUE = (0, 127, 255)
WHITE = (255, 255, 255)
RED = (190, 0, 0)
PINK = (255, 20, 147)
PURPLE = (128, 0, 145)
GREEN = (0, 190, 0)
YELLOW = (250, 250, 0)
ORANGE = (255, 128, 0) 


class AntsVisualizer:
    def __init__(self, ants_env: Ants, **kwargs: dict):
        self.ants_env = ants_env

        self.fps = kwargs['FPS']
        self.shade_strength = kwargs['SHADE_STRENGTH']
        self.show_chem_text_ph_0 = kwargs['SHOW_CHEM_TEXT_PH_0']
        self.show_chem_text_ph_1 = kwargs['SHOW_CHEM_TEXT_PH_1']
        self.cluster_font_size = kwargs['CLUSTER_FONT_SIZE']
        self.chemical_font_size = kwargs['CHEMICAL_FONT_SIZE']

        self.show_dirs_view = kwargs["show_dirs_view"]
        if self.show_dirs_view:
            self.dirs = self._get_dirs()
        self.show_ph_0_view = kwargs["show_ph_0_view"]
        self.show_ph_1_view = kwargs["show_ph_1_view"]
        
        self.first_gui = True
        self.screen = pygame.display.set_mode((self.ants_env.W_pixels, self.ants_env.H_pixels))
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.cluster_font = pygame.font.SysFont("arial", self.cluster_font_size)
        self.chemical_font = pygame.font.SysFont("arial", self.chemical_font_size)
        self.ph_pos_font = pygame.font.SysFont("arial", self.chemical_font_size * 2)

    def _get_dirs(self) -> np.ndarray:
        central = self.ants_env.wiggle_patches // 2
        sliding_window = []
        
        for i in range(self.ants_env.N_DIRS):
            tmp = []
            for j in range(self.ants_env.wiggle_patches):
                tmp.append((i + j) % self.ants_env.N_DIRS)
            sliding_window.append(tmp)
        
        sliding_window = sorted(sliding_window, key=lambda x: x[central])
        return np.array(sliding_window)

    def render(self, actions: np.ndarray) -> np.ndarray:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # window closed -> program quits
                pygame.quit()

        if self.first_gui:
            self.first_gui = False
            pygame.init()
            pygame.display.set_caption("ANTS")

        self.screen.fill(BLACK)
        # draw patches
        for p in self.ants_env.patches:
            if self.ants_env.patches[p]['chemical_0'] > self.ants_env.patches[p]['chemical_1']:
                chem_type = "chemical_0"
            elif self.ants_env.patches[p]['chemical_1'] > self.ants_env.patches[p]['chemical_0']:
                chem_type = "chemical_1"
            else:
                chem_type = "chemical_0"

            #chem_type = "chemical_0"
            chem = round(self.ants_env.patches[p][chem_type]) * self.shade_strength
            pygame.draw.rect(
                self.screen,
                (0, chem if chem <= 255 else 255, 0) if chem_type == "chemical_0" else (chem if chem <= 255 else 255, chem if chem <= 255 else 255, 0),
                pygame.Rect(
                    p[0] - self.ants_env.offset,
                    p[1] - self.ants_env.offset,
                    self.ants_env.patch_size,
                    self.ants_env.patch_size
                )
            )

            if self.show_chem_text_ph_0 and (
                not sys.gettrace() is None or self.ants_env.patches[p]['chemical_0'] >= self.ants_env.sniff_threshold
            ):
                text = self.chemical_font.render(str(round(self.ants_env.patches[p]['chemical_0'], 1)), True, GREEN)
                self.screen.blit(text, text.get_rect(center=p))
            
            if self.show_chem_text_ph_1 and (
                not sys.gettrace() is None or self.ants_env.patches[p]['chemical_1'] >= self.ants_env.sniff_threshold
            ):  
                text = self.chemical_font.render(str(round(self.ants_env.patches[p]['chemical_1'], 1)), True, YELLOW)
                self.screen.blit(text, text.get_rect(center=p))

            if p in self.ants_env.patches_food_1:
                text = self.ph_pos_font.render('F', True, WHITE)
                self.screen.blit(text, text.get_rect(center=p))
            elif p in self.ants_env.patches_food_2:
                text = self.ph_pos_font.render('F', True, WHITE)
                self.screen.blit(text, text.get_rect(center=p))
            elif p in self.ants_env.patches_food_3:
                text = self.ph_pos_font.render('F', True, WHITE)
                self.screen.blit(text, text.get_rect(center=p))
            elif p in self.ants_env.patches_nest:
                text = self.ph_pos_font.render('N', True, WHITE)
                self.screen.blit(text, text.get_rect(center=p))

        # draw learners
        for i, learner in enumerate(self.ants_env.learners.values()):
            if learner['flags'] != [0, 0, 0] and learner['flags'] != [0, 0, 1] and actions[i] == 3:
                learner_color = BLUE
            elif learner['flags'] != [0, 0, 0] and learner['flags'] != [0, 0, 1] and actions[i] != 3:
                learner_color = RED
            elif (learner['flags'] == [0, 0, 0] or learner['flags'] == [0, 0, 1]) and (actions[i] == 2 or actions[i] == 3):
                learner_color = RED
            elif (learner['flags'] == [0, 0, 0] or learner['flags'] == [0, 0, 1]) and actions[i] == 0:
                learner_color = ORANGE
            elif (learner['flags'] == [0, 0, 0] or learner['flags'] == [0, 0, 1]) and actions[i] == 1:
                learner_color = BLUE
            else:
                learner_color = PURPLE

            pygame.draw.circle(
                self.screen,
                learner_color,
                (learner['pos'][0], learner['pos'][1]),
                self.ants_env.turtle_size // 2
            )
            if learner['flags'] == [1, 0, 1]:
                text = self.cluster_font.render(str(i), True, SKY_BLUE)
                self.screen.blit(text, text.get_rect(center=learner['pos']))
            else:
                text = self.cluster_font.render(str(i), True, PINK)
                self.screen.blit(text, text.get_rect(center=learner['pos']))

            if self.show_dirs_view:
                if learner['pos'] in self.ants_env.fov_dirs.keys():
                    i, j = np.where(self.ants_env.fov_dirs[learner['pos']] == learner['dir'])
                    if i.size != 0 and j.size != 0:
                        new_dir = self.ants_env.fov_dirs[learner['pos']][-1][j].item()
                    else:
                        new_dir = learner['dir']
                    
                    x = np.any(
                        (
                            self.ants_env.fov[learner['pos']][new_dir] < self.ants_env.MIN_COORD[0]
                        ) | (
                            self.ants_env.fov[learner['pos']][new_dir] > self.ants_env.MAX_COORD[0]
                        ), 
                        axis=-1
                    )
                    y = np.where(x == False)[0]
                    view = self.ants_env.fov[learner['pos']][new_dir][y]
                    dirs = self.dirs[new_dir][y]
                else:
                    if len(self.ants_env.fov[learner["pos"]].shape) > 2:
                        view = self.ants_env.fov[learner["pos"]][learner["dir"]]
                        dirs = self.dirs[learner["dir"]]
                    else:
                        view = self.ants_env.fov[learner["pos"]]
                        dirs = self.dirs[4]
                
                for f, d in zip(view, dirs):
                    pygame.draw.rect(
                        self.screen,
                        YELLOW,
                        pygame.Rect(
                            f[0] - self.ants_env.offset,
                            f[1] - self.ants_env.offset,
                            self.ants_env.patch_size,
                            self.ants_env.patch_size
                        )
                    )
                    text = self.cluster_font.render(str(d), True, BLACK)
                    self.screen.blit(text, text.get_rect(center=f))

            if self.show_ph_0_view or self.show_ph_1_view:
                if learner['pos'] in self.ants_env.ph_fov_dirs.keys():
                    # Convenzione del momento:
                    # se l'agente ha una direzione perpendicolare all'ostacolo inverte la sua direzione,
                    # altrimenti viene approssimata con la posizione più vicina.
                    i, j = np.where(self.ants_env.ph_fov_dirs[learner['pos']] == learner['dir'])
                    if i.size != 0 and j.size != 0:
                        new_dir = self.ants_env.ph_fov_dirs[learner['pos']][-1][j].item()
                    else:
                        new_dir = learner['dir']
                    
                    x = np.any(
                        (
                            self.ants_env.ph_fov[learner['pos']][new_dir] < self.ants_env.MIN_COORD[0]
                        ) | (
                            self.ants_env.ph_fov[learner['pos']][new_dir] > self.ants_env.MAX_COORD[0]
                        ),
                        axis=-1
                    )
                    y = np.where(x == False)[0]
                    ph = self.ants_env.ph_fov[learner['pos']][new_dir][y]
                else:
                    if len(self.ants_env.ph_fov[learner["pos"]].shape) > 2:
                        ph = self.ants_env.ph_fov[learner["pos"]][learner["dir"]]
                    else:
                        ph = self.ants_env.ph_fov[learner["pos"]]
                
                for f in ph:
                    pygame.draw.rect(
                        self.screen,
                        WHITE,
                        pygame.Rect(
                            f[0] - self.ants_env.offset,
                            f[1] - self.ants_env.offset,
                            self.ants_env.patch_size,
                            self.ants_env.patch_size
                        )
                    )
                    if self.show_ph_0_view:
                        if self.ants_env.patches[tuple(f)]['chemical_0'] >= self.ants_env.sniff_threshold:
                            text = self.chemical_font.render(
                                str(round(self.ants_env.patches[tuple(f)]['chemical_0'], 1)),
                                True,
                                BLACK
                            )
                            self.screen.blit(text, text.get_rect(center=f))
                    elif self.show_ph_1_view:
                        if self.ants_env.patches[learner["pos"]]['chemical_1'] >= self.ants_env.sniff_threshold:
                            text = self.chemical_font.render(
                                str(round(self.ants_env.patches[tuple(f)]['chemical_1'], 1)),
                                True,
                                BLACK
                            )
                            self.screen.blit(text, text.get_rect(center=f))

        for p in self.ants_env.patches:
            if len(self.ants_env.patches[p]['turtles']) > 1:
                text = self.cluster_font.render(str(len(self.ants_env.patches[p]['turtles'])), True, WHITE)
                self.screen.blit(text, text.get_rect(center=p))

        self.clock.tick(self.fps)
        pygame.display.flip()

        return pygame.surfarray.array3d(self.screen)

    def close(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

def main():
    params = {
        "learners": 40,
        "actions": [
            "random-walk",
            "move-toward-chemical-0",
            "move-toward-chemical-1",
            "move-and-drop-chemical-1",
        ],
        "sniff_threshold": 0.9,
        "sniff_patches": 3, 
        "diffuse_area": 0.5,
        "diffuse_radius": 0,
        "wiggle_patches": 3,
        "lay_area": 1,
        "lay_amount": 5.0,
        "lay_amount_first": 1, 
        "lay_amount_min": 1.0,
        "ph_decay": 0.9,
        "evaporation": 0.95,
        "food_quantity": 1,
        "reward_type": "ind",
        "penalty": -0.1,
        "nest_penalty": -1.0,
        "ind_rew_scale_1": 1e2,
        "ind_rew_scale_2":3.5,
        "glob_rew_scale_1": 1e1,
        "glob_rew_scale_2": 2.0,
        "max_episode_ticks": 500,
        "W": 31,
        "H": 31,
        "PATCH_SIZE": 20,
        "TURTLE_SIZE": 16,
    }

    params_visualizer = {
        "FPS": 10,
        "SHADE_STRENGTH": 10,
        "SHOW_CHEM_TEXT_PH_0": False,
        "SHOW_CHEM_TEXT_PH_1": False,
        "CLUSTER_FONT_SIZE": 12,
        "CHEMICAL_FONT_SIZE": 8,
        "show_dirs_view": False,
        "show_ph_0_view": False,
        "show_ph_1_view": False,
    }

    from tqdm import tqdm

    EPISODES = 5
    SEED = 0
    np.random.seed(SEED)
    env = Ants(SEED, **params)
    env_vis = AntsVisualizer(env, **params_visualizer)
    ACTION_NUM = len(params["actions"])
    AGENTS_NUM = env.num_learners

    
    def policy(actions):
        return random.choice(actions)

    start_time = time.time()
    for _ in tqdm(range(1, EPISODES + 1), desc="Episode"):
        env.reset()
        ticks = 1
        while not env.done: 
            actions = np.array([-1 for _ in range(AGENTS_NUM)], dtype=np.int8)
            for agent in env.agent_iter(max_iter=AGENTS_NUM):
                agent_pos = env.learners[int(agent)]['pos']
                env.count_pos[env.pos_to_idx[agent_pos]] += 1
                
                obs, _, _, _, _ = env.last(agent)
                masked_actions = env.mask_actions(obs)
                action = policy(masked_actions)

                env.step(action)
                actions[int(agent)] = action
            
            env_vis.render(actions)
            ticks += 1
        
    print("Ticks: ", ticks)
    print("Total time = ", time.time() - start_time)
    env.close()

if __name__ == "__main__":
    main()