from typing import List, Dict
import json
import os
import gc
import numpy as np
import pickle
from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
# from vllm.distributed.parallel_state import destroy_model_parallel
import torch
from pipeline.jailbreak_config_generation import Config
from pipeline.submodules.evaluate_jailbreak import evaluate_completions_and_save_results_for_dataset
from pipeline.model_utils.model_factory import construct_model_base

model_path = 'google/gemma-2-9b-it'
contrastive_label = ['HHH', 'evil_confidant']
dataset_name = ['jailbreakbench', 'harmless']
save_path = 'D:\Data\jailbreak'


model_alias = os.path.basename(model_path)
artifact_path = os.path.join(save_path, "runs", "activation_pca", model_alias)
model_alias = os.path.basename(model_path)

cfg = Config(model_alias=model_alias, model_path=model_path,
             save_path=save_path, jailbreak_type="evil_confidant")
# model_base = construct_model_base(cfg.model_path)
save_path = artifact_path



evaluate_completions_and_save_results_for_dataset(cfg, dataset_name,
                                                  cfg.jailbreak_eval_methodologies,
                                                  contrastive_label,
                                                  save_path)