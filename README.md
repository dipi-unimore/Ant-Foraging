# Independent Learning of Stigmergic Communication in a Foraging Task

This document details the code, parameters, and configurations used to obtain the results described in **XXX** article.

---

## ⚙️ Installation

1. Make sure you have Python >= 3.8 installed.

2. Create a virtual environment:

    ```bash
    python -m venv path_to_new_virtual_env 
    source path_to_new_virtual_env/bin/activate
    ```

3. Clone this repository:

    ```bash
    git clone Ant-Foraging
    cd Ant-Foraging
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

**Operating system**: The code was tested on Ubuntu 22.04, CPU only. 

---

## 📂 Project Structure

```plaintext
Ant-Foraging/
├── agents                      # Algorithms folder
│   ├── IQLearning              # Indipendent Q-Learning implementation
│   │   └── config              # Algorithm configuration files
│   ├── NoLearning              # Deterministic policy implementation
│   └── utils                   # Utility functions
└── environments                # Multi-agent environments
    └── ants_env                # Ants environment
        └── config              # Env configuration files 
```

---

## 🚀 Running the Code

### IQLearning
    
The main script is `ants_iql.py`, which accepts the following command-line arguments:

| Argument            | Type    | Default value | Description                                                           |
|---------------------|---------|---------------|-----------------------------------------------------------------------|
| `--train` | bool | False | If `True`, training of the agents will be performed, else evaluation. |
| `--random_seed` | int | 42 | Change the default random seed for reproducibility. |
| `--qtable_path` | str | Empty String | Path to a `.npy` file for loading the Q-table to perform evaluation. |
| `--fixed_foods` | bool | False | This parameter is for testing purposes only. If it is set to `True`, the locations of food sources remain unchanged; otherwise, their locations are randomized. |
| `--print_metrics` | int | 30 | Metrics printing frequency. | 
| `--render` | bool | False | If `True`, renders the environment visually. |

**Example: Training run**

```bash
python ants_iql.py --train True --random_seed 99 
```

The **QTable** will be automatically put in the `./runs/weights` folder. 

**Example: Evaluation run**

```bash
python ants_iql.py --random_seed 99 --qtable_path ./runs/weights/file_name.npy --render True
```

### Deterministic Policy
    
The main script is `ants_baseline.py`:

**Run example**

```bash
python ants_baseline.py
```

---

## ⚙️ Key Parameters

### Environment

| Parameter                 | Values                         | Description                                  |
|---------------------------|-------------------------------------------|-----------------------------------|
| `World-size` | (**31x31**) | Size of the grid world where agents move. |
| `Learners` | **40** | Number of agents. |
| `Sniff-threshold`| **0.9** | Minimum amount of pheromone that can be <br> smelled by an agent. |
| `Sniff-patches` | **3** | Number of 1-hop neighboring patches in which the agent <br> can smell the pheromone. |
| `Wiggle-patches` | **3** | Number of 1-hop neighboring patches the agent can move <br> randomly through. |
| `Diffuse-area` | [**0.5** for test, **0.83** for training] | Standard deviation value of the Gaussian function used to <br> spread the pheromone in the environment. |
| `Diffuse-radius` | **0.0** | Radius of the Gaussian function used to spread <br> the pheromone in the environment. |
| `Lay-area` | **1** | Number of patches in which the pheromone is released. |
| `Lay-amount` | **5** | Amount of pheromone deposited evenly in `Lay-area`. |
| `Lay-amount-first` | **1** | `Lay-amount` multiplier. Only applies the first time the agent releases the pheromone after finding food. |
| `Lay-amount-min` | **1.0** | The minimum level below which the pheromone intensity cannot fall. It is used as a pheromone intensity until the agent finds food.|
| `Ph-decay` | **0.9** | The value by which the pheromone intensity is multiplied after the agent has found food. |
| `Evaporation-rate` | **0.95** | Amount of pheromone not evaporating in the environment. |
| `Food-quantity` | **5** | Amount of food per patch. |

### Learning

| Parameter                 | Values                 | Description                   |
|---------------------------|------------------------|-------------------------------|
| `Reward-type` | [**ind**, **glob**, **mix**] | 3 type of reward: `ind` (individual reward), `glob` (global reward) and `mix` (individual + global). |
| `Penalty` | **-0.1** | Base penalty imposed for failing to collect food outside the nest. |
| `Nest-penalty` | **-1** | Base penalty imposed for failing to collect food inside the nest. |
| `Ind-rew-scale-1` | **100** | Used to scale the individual reward. |
| `Ind-rew-scale-2` | **3.5** | Used to scale the individual reward when the agent returns to the nest with food. |
| `Glob-rew-scale-1` | **10** | Used to scale the global reward when the agent returns to the nest with food. |
| `Glob-rew-scale-2` | **2.5** | Used to scale the global reward when the agent returns to the nest with food. |
| `Max-episode-ticks` | [**500** for training, **1000** for test] | Learning episode duration in simulation ticks. |
| `Episodes` | **3000** | Number of learning episodes. |
| `Learning-rate` ($\alpha$) | **0.01** | Magnitude of Q-values updates. |
| `Discount-factor` ($\gamma$) | **0.99** | How much future rewards are given value. |
| `Epsilon-init` ($\epsilon_{init}$) | **1.0** | Initial exploration rate. |
| `Epsilon-min`  ($\epsilon_{min}$) | **0.0** | Minimum value of epsilon. |
| `Epsilon-decay` ($\lambda$) | **0.995** | How much epsilon lowers after each action, <br> it goes from ($\epsilon_{init}$) to ($\epsilon_{min}$). |

---

## 🛠️ Configuration Files

The following .json configuration files are used to manage the experiment's parameters:

| File Name |	Purpose |
|-----------|---------|
| `/environments/ants_env/config/env-params.json` | Defines the environment settings. |
| `/environments/ants_env/config/env_visualizer-params.json` | Controls the rendering configuration for visualizing the environment. |
| `/agent/IQLearning/config/learning-params.json` |	Contains learning-related parameters such as learning rate, epsilon decay, etc. |
| `/agent/IQLearning/config/logger-params.json` | Configures the logging behavior and export mode. |

---

## 📈 Evaluation Metrics

All evaluation metrics described in the paper are automatically logged in `/runs/train` (for training) and in `/runs/eval`(for evaluation).

---

## 💾 Reproducibility

Ours paper results presents the average result of 10 identical experiments. The random seeds we used: `[10, 20, 30, 40, 50, 60, 70 , 80, 90, 100]`.

---

## 📚 Citation

If you use this codebase in your research, please cite the following article:

> **XXX** 
> Authors: Davide Borghi, Stefano Mariani, and Franco Zambonelli  
> XXX