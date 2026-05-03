import datetime
from typing import Optional
import time

import numpy as np
from environments.ants_env.ants import Ants, AntsVisualizer
from agents.NoLearning.baseline import policy, plot_metrics

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
        "lay_amount": 2.0,
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
        "SHOW_CHEM_TEXT_PH_0": True,
        "SHOW_CHEM_TEXT_PH_1": True,
        "CLUSTER_FONT_SIZE": 12,
        "CHEMICAL_FONT_SIZE": 8,
        "show_dirs_view": False,
        "show_ph_0_view": False,
        "show_ph_1_view": False,
    }

    from tqdm import tqdm

    EPISODES = 10
    SEED = 100
    np.random.seed(SEED)
    env = Ants(SEED, eval=True, fixed_food=True, **params)
    env_vis = AntsVisualizer(env, **params_visualizer)
    ACTION_NUM = len(params["actions"])
    AGENTS_NUM = env.num_learners 


    ticks = np.zeros(EPISODES)
    rewards_x_ep = np.zeros(EPISODES)
    actions_x_ep = np.zeros((EPISODES, ACTION_NUM))
    total_food = np.zeros(EPISODES)
    food_counts = np.zeros(EPISODES)
    #time_food_1 = []
    #time_food_2 = []
    #time_food_3 = []

    start_time = time.time()
    for ep in tqdm(range(1, EPISODES + 1), desc="Episode"):
        env.reset()
        tick = 1
        rewards = np.zeros(AGENTS_NUM)
        track_actions = np.zeros(ACTION_NUM)
        
        while not env.done: 
            actions = np.array([-1 for _ in range(AGENTS_NUM)], dtype=np.int8)
            for agent in env.agent_iter(max_iter=AGENTS_NUM):
                agent_pos = env.learners[int(agent)]['pos']
                env.count_pos[env.pos_to_idx[agent_pos]] += 1

                observation, reward, _ , _, _ = env.last(agent)
                id = env.convert_observation(observation)
                
                action = policy(env.learners[int(agent)], observation, env.sniff_patches, env.sniff_threshold)
                env.step(action)

                rewards[int(agent)] += round(reward, 4)
                track_actions[action] += 1
                actions[int(agent)] = action
            
            env_vis.render(actions)
            tick += 1

        ticks[ep - 1] = tick
        rewards_x_ep[ep - 1] = round((rewards.sum() / tick) / AGENTS_NUM, 4)
        actions_x_ep[ep - 1] = (track_actions / tick) / AGENTS_NUM
        total_food[ep - 1] = env.food_x_agent.sum().item()
        food_counts[ep - 1] = sum(list(env.food_counts.values()))
        #time_food_1.append(env.time_food_1)
        #time_food_2.append(env.time_food_2)
        #time_food_3.append(env.time_food_3)

    print("Total time = ", time.time() - start_time)
    
    plot_metrics(
        EPISODES,
        rewards_x_ep,
        actions_x_ep.round(2),
        ticks,
        total_food,
        food_counts,
        #time_food_1,
        #time_food_2,
        #time_food_3,
        params['actions']
    )

    date = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    np.save(f"agents/NoLearning/baseline_count_pos_{date}.npy", env.count_pos)
    env.close()

if __name__ == "__main__":
    main()