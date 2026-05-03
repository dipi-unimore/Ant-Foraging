import itertools
import random
import datetime
import numpy as np
from tqdm import tqdm
from typing import Union, List, Tuple, Dict

class IQL:

    def __init__(
        self,
        env,
        env_vis,
        qtable_path: str,
        base_logger,
        other_logger,
        **learning_params: Dict
    ) -> None:
        self.env = env
        self.env_vis = env_vis
        self.base_logger = base_logger
        self.other_logger = other_logger

        self.alpha = learning_params["alpha"]
        self.gamma = learning_params["gamma"]
        self.epsilon = learning_params["epsilon"]
        self.epsilon_min = learning_params["epsilon_min"]
        self.decay_type = learning_params["decay_type"]
        self.decay = learning_params["decay"]
        self.train_episodes = learning_params["train_episodes"]
        self.test_episodes = learning_params["test_episodes"]
        
        if len(qtable_path) != 0:
            self.qtable = self.base_logger.load_model()
        else:
            self.qtable = np.zeros([self.env.num_learners, self.env.POSSIBLE_STATES, self.env.POSSIBLE_ACTIONS])

    def _get_metrics(self) -> Tuple[Dict, Dict, Dict]:
        actions_dict = {
            str(ep): {
                str(ac): 0 
                for ac in range(self.env.POSSIBLE_ACTIONS)
            } for ep in range(1, self.train_episodes + 1)
        } 
        
        action_dict = {
            str(ep): {
                str(ag): {
                    str(ac): 0 
                    for ac in range(self.env.POSSIBLE_ACTIONS)
                } for ag in range(self.env.num_learners)
            } for ep in range(1, self.train_episodes + 1)
        }
        
        reward_dict = {
            str(ep): {
                str(ag): 0 
                for ag in range(self.env.num_learners)
            }
            for ep in range(1, self.train_episodes + 1)
        }

        return action_dict, actions_dict, reward_dict

    def train(self, print_metrics) -> datetime.timedelta:
        action_dict, actions_dict, reward_dict = self._get_metrics()
        old_s = {}
        old_a = {}
        #actions = [4, 5]
        AGENTS_NUM = self.env.num_learners
        self.other_logger.train_log_every = 500

        # TRAINING
        print("Start training...\n")
        train_start = datetime.datetime.now()

        for ep in tqdm(range(1, self.train_episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
            self.env.reset()
            tick = 1
            
            while not self.env.done: 
                actions = np.array([-1 for _ in range(AGENTS_NUM)], dtype=np.int8)
                nest = []
                food = []
                for agent in self.env.agent_iter(max_iter=AGENTS_NUM):
                    obs, reward, _, _, _ = self.env.last(agent)
                    state = self.env.convert_observation(obs)
                    masked_actions = self.env.mask_actions(obs)
                    
                    if ep == 1 and tick == 1:
                        action = np.random.choice(masked_actions, 1).item()
                    else:
                        # QTable update
                        old_value = self.qtable[int(agent), old_s[agent], old_a[agent]]
                        next_max = np.max(self.qtable[int(agent), state])  # QUESTION: was with [action] too
                        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                        self.qtable[int(agent), old_s[agent], old_a[agent]] = new_value
                        
                        # next_action
                        if random.uniform(0, 1) < self.epsilon:
                            action = np.random.choice(masked_actions, 1).item()
                        else:
                            idx = np.argmax(self.qtable[int(agent)][state][masked_actions])
                            action = masked_actions[idx].item()

                    self.env.step(action)
                    
                    actions[int(agent)] = action
                    nest.append(bool(int(obs[-1])))
                    food.append(bool(int(obs[-2])))
                    
                    old_s[agent] = state
                    old_a[agent] = action

                    actions_dict[str(ep)][str(action)] += 1
                    action_dict[str(ep)][str(agent)][str(action)] += 1
                    reward_dict[str(ep)][str(agent)] += round(reward, 2)
                
                if self.env_vis != None:
                    self.env_vis.render(actions)
                
                if ep % self.other_logger.train_log_every == 0:
                    data = [ep, tick]
                    data.extend(actions.tolist())
                    data.extend(nest)
                    data.extend(food)
                    self.other_logger.load_value(data)
                
                tick += 1
            
            if self.decay_type == "log":
                self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)
            elif self.decay_type == "linear":
                self.epsilon = max(self.epsilon - self.decay, self.epsilon_min)
            
            total_food = self.env.food_nest

            if ep % self.base_logger.train_log_every == 0:
                value = [ep, tick]
                avg_rew = round((sum(reward_dict[str(ep)].values()) / tick) / AGENTS_NUM, 4)
                value.append(avg_rew)
                value.extend(np.array(list(reward_dict[str(ep)].values())).round(2).tolist())
                value.append(total_food)
                value.extend(self.env.food_x_agent.tolist())
                value.extend(list(actions_dict[str(ep)].values()))
                tmp = [list(v.values()) for v in action_dict[str(ep)].values()]
                value.extend(list(itertools.chain(*tmp)))
                eps = round(self.epsilon, 4)
                value.append(eps)
                
                self.base_logger.load_value(value)

                if ep % print_metrics == 0:
                    print("\nMetrics ")
                    print(" - ticks: ", tick)
                    print(" - avg_reward: ", avg_rew)
                    print(" - food: ", total_food)
                    print(" - epsilon: ", eps)
                    print(" - alpha:", self.alpha)
                    print(" - lay_amount_food:", self.env.lay_amount_food)
            
        self.base_logger.empty_table()
        self.other_logger.empty_table()
        self.env.close()
        
        if self.env_vis != None:
            self.env_vis.close()

        train_end = datetime.datetime.now()
        print("Training finished!\n")
        
        computation_time = train_end - train_start
        print(f"Training time: {computation_time}\n")
        
        print("Now saving the model...\n")
        self.base_logger.save_model(self.qtable)
        print("Model saved.")

        return computation_time

    def eval(self) -> datetime.timedelta:
        action_dict, actions_dict, reward_dict = self._get_metrics()
        AGENTS_NUM = self.env.num_learners
        
        print("Start testing...\n")
        test_start = datetime.datetime.now()
        
        for ep in tqdm(range(1, self.test_episodes + 1), desc="EPISODES", colour='red', leave=False):
            self.env.reset()
            tick = 1
            
            while not self.env.done: 
                actions = np.array([-1 for _ in range(AGENTS_NUM)], dtype=np.int8)
                for agent in self.env.agent_iter(max_iter=AGENTS_NUM):
                    agent_pos = self.env.learners[int(agent)]['pos']
                    self.env.count_pos[self.env.pos_to_idx[agent_pos]] += 1

                    obs, reward, _, _, _ = self.env.last(agent)
                    state = self.env.convert_observation(obs)
                    
                    masked_actions = self.env.mask_actions(obs)
                    idx = np.argmax(self.qtable[int(agent)][state][masked_actions])
                    action = masked_actions[idx].item()
                    
                    self.env.step(action)
                    actions[int(agent)] = action
                    
                    actions_dict[str(ep)][str(action)] += 1
                    action_dict[str(ep)][str(agent)][str(action)] += 1
                    reward_dict[str(ep)][str(agent)] += round(reward, 2)
                    
                if self.env_vis != None:
                    self.env_vis.render(actions)

                tick += 1
            
            total_food = self.env.food_x_agent.sum().item()
            food_counts = sum(list(self.env.food_counts.values()))

            if ep % self.base_logger.test_log_every == 0:
                value = [ep, tick]
                value.append(food_counts)
                value.append(total_food)
                value.extend(self.env.food_x_agent.tolist())
                value.append(self.env.time_food_1)
                value.append(self.env.time_food_2)
                value.append(self.env.time_food_3)
                value.extend(list(actions_dict[str(ep)].values()))
                tmp = [list(v.values()) for v in action_dict[str(ep)].values()]
                value.extend(list(itertools.chain(*tmp)))
                self.base_logger.load_value(value)
            
        self.base_logger.log_positions(self.env.count_pos)

        self.base_logger.empty_table()
        self.env.close()
        
        if self.env_vis != None:
            self.env_vis.close()

        test_end = datetime.datetime.now()
        print("Testing finished!")
        
        computation_time = test_end - test_start
        print(f"Test time: {computation_time}\n")

        return computation_time