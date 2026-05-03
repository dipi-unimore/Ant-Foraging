import datetime
import errno
import gc
import os
import json
from collections import deque
from typing import Union, List, Tuple, Dict

import pandas as pd


class Logger:

    def __init__(
        self, 
        curdir: str, 
        metrics: List,
        train: bool, 
        weights_file: str = "",
        output_file_append: str = "",
        **log_params: Dict, 
    ) -> None:
        self.train_log_every = log_params["train_log_every"]
        self.test_log_every = log_params["test_log_every"]
        self.deep_algo = log_params["deep_algorithm"] 
        self.buffer_size = log_params["buffer_size"]
        
        self.OUTPUT_FILE_EXTENSION = ".csv"
        self.WEIGHTS_FILE_EXTENSION = ".npy" if self.deep_algo == False else ".pth"
        self.PARAMS_FILE_EXTENSION = ".txt"

        self.curdir = curdir
        self.train = train
        self.output_file_append = output_file_append
        
        self.metrics = tuple(metrics)
        self.table = pd.DataFrame(columns=self.metrics)

        self.mode = "train" if self.train else "eval"
        time_now = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

        self.base_dir = os.path.join(self.curdir, "runs/" + self.mode)
        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir)

        self.output_dir = os.path.join(self.base_dir, self.mode + "_" + time_now)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if len(self.output_file_append) != 0:
            self.filename = log_params[self.mode + "_output_file"].replace("-", "_") + '_' + time_now + '_' + self.output_file_append + self.OUTPUT_FILE_EXTENSION
        else:
            self.filename = log_params[self.mode + "_output_file"].replace("-", "_") + '_' + time_now + self.OUTPUT_FILE_EXTENSION
        self.output_file = os.path.join(self.output_dir, self.filename)

        self.params_filename = log_params[self.mode + "_params_file"] + '_' + time_now + self.PARAMS_FILE_EXTENSION
        self.params_file = os.path.join(self.output_dir, self.params_filename)
        
        self.weights_dir = os.path.join(curdir, "runs/weights")
        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)
        
        if self.train:
            if len(weights_file) == 0:
                weights_filename = log_params[self.mode + "_weights_file"] + '_' + time_now + self.WEIGHTS_FILE_EXTENSION 
                self.weights_file = os.path.join(self.weights_dir, weights_filename)
            else:
                self.weights_file = weights_file
        else:
            if len(weights_file) == 0:
                if not os.path.isdir(self.weights_dir):
                    raise FileNotFoundError(errno.ENOENT, "No such directory", self.weights_dir)
                
                self.weights_file = self._get_weight_path()
            else:
                self.weights_file = weights_file
    
    def _get_weight_path(self) -> str:
        if not os.path.isdir(self.weights_dir):
            raise FileNotFoundError(errno.ENOENT, "No such directory found", self.weights_dir)
        
        weights_filename = sorted([
            f for f in os.listdir(self.weights_dir) if os.path.isfile(os.path.join(self.weights_dir, f)) and f.endswith(self.WEIGHTS_FILE_EXTENSION)
        ])[-1]
        
        if len(weights_filename) == 0:
            if self.deep_algo:
                raise FileNotFoundError(errno.ENOENT, "No such weights file (.pth) found in", self.weights_dir)
            else:
                raise FileNotFoundError(errno.ENOENT, "No such weights file (.npy) found in", self.weights_dir)
        
        weights_file = os.path.join(self.weights_dir, weights_filename)
        return weights_file
    
    def _add_rows(self, vals) -> None:
        tmp = pd.DataFrame(vals, columns=self.metrics)
        if self.table.shape[0] == 0:
            self.table = self.table.combine_first(tmp)
        else:
            self.table = pd.concat([self.table, tmp], ignore_index=True)
    
    def _delete_table(self) -> None:
        self.table = None
        gc.collect()

    def _reinit(self) -> None:
        self._delete_table()
        self.table = pd.DataFrame(columns=self.metrics)

    def load_values(self, values: Union[List, Tuple]) -> None:
        assert(isinstance(values, list) or isinstance(values, tuple)), "Error: values must be of type list or tuple!"
        values = deque(values)
        while(len(values) != 0):
            # Check if full
            quantity = self.buffer_size - self.table.shape[0]
            if quantity == 0:
                flag = self._write_to_csv()
                if flag:
                    self._reinit()
            else:
                if quantity >= len(values):
                    tmp = [values.popleft() for _ in range(len(values))]
                else:
                    tmp = [values.popleft() for _ in range(quantity)]
                self._add_rows(tmp)

    def load_value(self, value: Union[List, Tuple]) -> None:
        assert(isinstance(value, list) or isinstance(value, tuple)), "Error: value must be of type list or tuple!"
        quantity = self.buffer_size - self.table.shape[0]
        if quantity == 0:
            flag = self._write_to_csv()
            if flag:
                self._reinit()
        self._add_rows([value])
    
    def _write_to_csv(self) -> bool:
        if os.path.isfile(self.output_file): # check se il file esiste
            with open(self.output_file, 'a') as f:
                self.table.to_csv(f, header=False, sep=',', index=False)
        else: # check se il file non esiste
            self.table.to_csv(self.output_file, sep=',', index=False)
        return True
    
    def write_params(
        self,
        env_params: Dict,
        learning_params: Dict,
        log_params: Dict,
        computation_time: float,
        train: bool
    ) -> None:
        with open(self.params_file, 'w') as f:
            f.write(f"{json.dumps(env_params, indent=2)}\n")
            f.write("----------\n")
            f.write(f"{json.dumps(learning_params, indent=2)}\n")
            f.write("----------\n")
            f.write(f"{json.dumps(log_params, indent=2)}\n")
            if not train:
                f.write(f"weights_file = {self.weights_file}\n")
            f.write("----------\n")
            if train:
                f.write(f"Training time: {computation_time}\n")
            else:
                f.write(f"Testing time: {computation_time}\n")
    
    def empty_table(self) -> None:
        if self.table.shape[0] > 0:
            self._write_to_csv()
        self._delete_table()

    def load_model(self):
        if self.deep_algo:
            from torch import load
            return load(self.weights_file)
        else:
            from numpy import load
            return load(self.weights_file)
    
    def save_model(self, weights) -> None:
        if self.deep_algo:
            from torch import save
            save(weights, self.weights_file)
        else:
            from numpy import save
            save(self.weights_file, weights)
    
    def log_positions(self, count_pos) -> None:
        from numpy import save
        
        time = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        save(f"{self.output_dir}/count_pos_{time}.npy", count_pos)


def get_base_metrics(food_sources: int, learners: int, actions: List, train: bool, deep_algo: bool) -> List:
    metrics = ["Episode", "Tick"]

    if train:
        metrics.append("Avg reward X episode") 
        for l in range(learners):
            metrics.append(f"agent-{l}_reward")

    if not train:
        metrics.append("Food counts")

    metrics.append("Total food")
    for l in range(learners):
        metrics.append(f"agent-{l}_food")

    if not train:
        for i in range(food_sources):
            metrics.append(f"time-food_{i + 1}")

    for a in actions:
        metrics.append(a)
    for l in range(learners):
        for a in actions:
            metrics.append(f"(learner {l})-{a}")

    if train:
        metrics.append("Epsilon")
        if deep_algo:
            metrics.append("Loss")
            #metrics.append("Learning rate")

    return metrics
 
def get_other_metrics(learners: int) -> List:
    metrics = ["Episode", "Tick"]
    
    for a in range(learners):
        metrics.append(f"agent-{a}_action")
    
    for a in range(learners):
        metrics.append(f"agent-{a}_nest")
    
    for a in range(learners):
        metrics.append(f"agent-{a}_food")
    
    return metrics

def create_loggers(
    curdir: str,
    food_sources: int, 
    learners: int,
    actions: List,
    train: bool,
    weights_file: str,
    log_params: Dict
) -> Tuple[Logger, Logger]:
    base_metrics = get_base_metrics(food_sources, learners, actions, train, log_params["deep_algorithm"])
    other_metrics = get_other_metrics(learners) 

    base_logger = Logger(
        curdir,
        base_metrics,
        train,
        weights_file,
        **log_params,
    )
    
    other_logger = Logger(
        curdir,
        other_metrics,
        train,
        output_file_append="other",
        **log_params
    )

    return base_logger, other_logger