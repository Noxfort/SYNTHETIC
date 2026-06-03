# SYNTHETIC  - An AI-Orchestrated Engine for Multi-Modal Traffic Scenario Synthesis
# Copyright (C) 2026 Noxfort Systems 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SOFTWARE.
#
# File: optimizer/callbacks.py
# Author: Gabriel Moraes
# Date: 2025-11-27

import optuna

class PruningCallback:
    """
    Custom callback to integrate PyTorch training loops with Optuna's Pruning mechanism.
    It checks intermediate results and decides whether to stop the trial early.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val_loss"):
        """
        Args:
            trial (optuna.trial.Trial): The current optimization trial.
            monitor (str): The metric name to monitor (mostly 'loss' for GANs).
        """
        self.trial = trial
        self.monitor = monitor

    def check_pruned(self, current_step: int, current_score: float) -> None:
        """
        Reports the current score to Optuna and checks if the trial should be pruned.
        
        Args:
            current_step (int): The current epoch or iteration number.
            current_score (float): The value of the monitored metric (e.g., Loss).
        
        Raises:
            optuna.exceptions.TrialPruned: If Optuna decides this trial is not promising.
        """
        # Report the current value to Optuna
        self.trial.report(current_score, current_step)

        # Check if the trial should be pruned based on the configured Pruner (e.g., MedianPruner)
        if self.trial.should_prune():
            message = f"Trial pruned at step {current_step} with {self.monitor}={current_score:.4f}"
            print(f"[AutoML] {message}")
            raise optuna.exceptions.TrialPruned(message)

class EarlyStopping:
    """
    Standard Early Stopping mechanism independent of Optuna.
    Stops training if the loss does not improve after a set number of epochs (patience).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_loss: float):
        """
        Call at the end of each epoch to check status.
        
        Args:
            current_loss (float): The validation loss of the current epoch.
        """
        if self.best_score is None:
            self.best_score = current_loss
        elif current_loss > self.best_score - self.min_delta:
            # Loss did not decrease significantly
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("[EarlyStopping] Patience limit reached. Stopping training.")
        else:
            # Loss improved
            self.best_score = current_loss
            self.counter = 0