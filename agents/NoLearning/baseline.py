from typing import Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from environments.ants_env.ants import Ants, AntsVisualizer

def policy(turtle: Dict, obs: np.ndarray, patches: int, th: float):
    max_ph_pos = obs[:patches].argmax()
    max_ph = obs[max_ph_pos]
    
    if max_ph >= th:
        action = 1
    else:
        action = 0

    if turtle['flags'] == [1, 0, 0] or turtle['flags'] == [1, 0, 1]:
        action = 3
    
    return action

def plot_metrics(
        ep: int,
        rewards_x_ep: np.ndarray,
        actions_x_ep: np.ndarray,
        ticks: np.ndarray,
        total_food: np.ndarray,
        food_counts: np.ndarray,
        #time_food_1: List,
        #time_food_2: List,
        #time_food_3: List,
        actions_name: List
    ):

    data = {
        "Ticks": ticks,
        "Avg_Reward": rewards_x_ep,
        "Total_food": total_food,
        "Food_counts": food_counts,
    }
    data.update({actions_name[i]: actions_x_ep[:, i] for i in range(actions_x_ep.shape[1])})
    df = pd.DataFrame(data)
    df.to_csv("agents/NoLearning/baseline_metrics.csv", sep=',', index=False)
    x = np.array([e for e in range(ep)])
    
    avg_reward = rewards_x_ep.mean()
    y = np.array([avg_reward for _ in range(ep)])
    fig = plt.figure(figsize=(10, 5), dpi=200)
    plt.ylabel(f"Avg Reward = {round(avg_reward, 4)}, Std = {round(rewards_x_ep.std(), 4)}")
    plt.scatter(x, rewards_x_ep, s=4, linewidth=1.0, color='#648FFF')
    plt.plot(x, y, label="mean", marker='x', markersize=.5, linewidth=.5, color='#DC267F')
    fig.tight_layout()
    plt.savefig("agents/NoLearning/Avg_Reward")
    
    avg_tick = ticks.mean()
    y = np.array([avg_tick for _ in range(ep)])
    fig = plt.figure(figsize=(10, 5), dpi=200)
    plt.ylabel(f"Avg Tick = {round(avg_tick, 4)}, Std = {round(ticks.std(), 4)}")
    plt.scatter(x, ticks, s=4, linewidth=1.0, color='#648FFF')
    plt.plot(x, y, label="mean", marker='x', markersize=.5, linewidth=.5, color='#DC267F')
    fig.tight_layout()
    plt.savefig("agents/NoLearning/Avg_Ticks")

    fig = plt.figure(figsize=(10, 5), dpi=200)
    y = np.array([food_counts.mean() for _ in range(ep)])
    plt.ylabel(f"Food counts = {round(food_counts.mean(), 4)}, Std = {round(food_counts.std(), 4)}")
    plt.plot(x, food_counts, marker='x', markersize=.5, linewidth=.5, color='#DC267F')
    plt.plot(x, y, marker='x', markersize=.5, linewidth=.5, color='blue')
    fig.tight_layout()
    plt.savefig("agents/NoLearning/Food_counts")
    
    fig = plt.figure(figsize=(10, 5), dpi=200)
    y = np.array([total_food.mean() for _ in range(ep)])
    plt.ylabel(f"Total food = {round(total_food.mean(), 4)}, Std = {round(total_food.std(), 4)}")
    plt.plot(x, total_food, marker='x', markersize=.5, linewidth=.5, color='#DC267F')
    plt.plot(x, y, marker='x', markersize=.5, linewidth=.5, color='blue')
    fig.tight_layout()
    plt.savefig("agents/NoLearning/Total_food")
    
    fig = plt.figure(figsize=(10, 5), dpi=200)
    for a in range(len(actions_name)):
        plt.plot(
            x,
            actions_x_ep[:, a] * 100,
            label=actions_name[a],
        )
    plt.yticks([i for i in range(0, 110, 10)])
    plt.title("Baseline normalized actions")
    plt.legend()
    plt.savefig("agents/NoLearning/Actions")