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
from pipeline.jailbreak_config_generation_intervention import Config
from pipeline.submodules.evaluate_jailbreak import evaluate_completions_and_save_results_for_dataset
from pipeline.model_utils.model_factory import construct_model_base

model_path = 'google/gemma-2-2b-it'
intervention = 'skip_connection_attn'
contrastive_label = ['HHH', 'BREAK']
dataset_name = ['harmful', 'harmless']
save_path = 'D:\Data\jailbreak'


model_alias = os.path.basename(model_path)
artifact_path = os.path.join(save_path, "runs", "activation_pca", model_alias)
model_alias = os.path.basename(model_path)
source_layer = 10
target_layer_s = 10
target_layer_e = None

cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path,
             intervention=intervention,
             source_layer=source_layer,
             target_layer_s=target_layer_s, target_layer_e=target_layer_e)
model_base = construct_model_base(cfg.model_path)
n_layers = model_base.model.config.num_hidden_layers
save_path = artifact_path + os.sep + intervention + os.sep

for layer in range(n_layers):
    if 'skip_connection' in intervention:
        source_layer = 0
        target_layer_s = layer
        target_layer_e = layer + 1

    else:
        source_layer = layer
        target_layer_s = layer
        target_layer_e = None

    evaluate_completions_and_save_results_for_dataset(cfg, dataset_name,
                                                      cfg.jailbreak_eval_methodologies,
                                                      contrastive_label,
                                                      save_path,
                                                      source_layer=source_layer,
                                                      target_layer_s=target_layer_s,
                                                      target_layer_e=target_layer_e,
                                                      few_shot=None)