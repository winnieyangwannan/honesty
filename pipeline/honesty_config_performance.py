
import os

from dataclasses import dataclass
from typing import Tuple
from typing import List

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

    model_alias: str
    model_path: str
    save_path: str
    checkpoint: int
    n_train: int = 100
    n_test: int = 32
    data_category: str = "facts"
    batch_size: int = 16
    source_layer: int = 10
    intervention: str = "no_intervention"
    target_layer: int = 10
    max_new_tokens: int = 100
    # for generation_trajectory
    dataset_id = [3, 4]
    layer_plot: int = 16
    sub_modules: str = "residual"

    def artifact_path(self) -> str:
        save_path = self.save_path
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(save_path, "runs", "activation_pca", self.model_alias)