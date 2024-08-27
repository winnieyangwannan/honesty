
import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # model_alias: str
    # model_path: str
    # n_train: int = 64
    # n_test: int = 32
    # data_category: str = "facts"
    # batch_size = 16
    # source_layer = 14
    # intervention = "direction ablation"
    # target_layer: int = None
    model_path: str
    model_alias: str
    task_name: str
    save_path: str
    save_name: str
    contrastive_label: str
    source_layer: int
    intervention: str
    target_layer: int
    hook_name: str
    n_train: int = 100
    n_test: int = 100
    batch_size: int = 10
    max_new_tokens: int = 100
    steering_strength: int = 1

    # for generation_trajectory

    def artifact_path(self) -> str:
        save_path = self.save_path
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(save_path, "runs", "activation_pca", self.model_alias)