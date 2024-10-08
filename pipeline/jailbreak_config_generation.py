
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

    model_alias: str
    model_path: str
    save_path: str
    jailbreak_type: str
    filter_train: bool = True
    filter_val: bool = True
    few_shot: int = None
    n_train: int = 50
    n_eval: int = 50
    data_category: str = "facts"
    batch_size: int = 1
    max_new_tokens: int = 400
    intervention: str = "no_intervention"
    evaluation_datasets: Tuple[str] = ("jailbreakbench", "harmless")
    # evaluation_datasets: Tuple[str] = ("jailbreakbench",)
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching",)
    # evalution_persona: Tuple[str] = ("BREAK",)

    # for generation_trajectory
    dataset_id: list[int] = 3

    def artifact_path(self) -> str:
        save_path = self.save_path
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(save_path, "runs", "activation_pca", self.model_alias)