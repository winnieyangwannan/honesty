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


import argparse


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='google/gemma-2-9b-it')
    parser.add_argument('--save_path', type=str, required=False, default='D:\Data\jailbreak')
    parser.add_argument('--prompt_type', type=str, required=False, default=16)
    parser.add_argument('--layer_plot', type=int, required=False, default=41)
    parser.add_argument('--contrastive_type', metavar='N', type=str, nargs='+',
                        help='a list of strings')

    return parser.parse_args()


def run_pipeline(model_path, save_path,
                 data_type=["jailbreakbench", "harmless"],
                 contrastive_type=['evil_confidant', 'AIM'],
                 ):
    # dataset_name = ['jailbreakbench', 'harmless']

    model_alias = os.path.basename(model_path)
    artifact_path = os.path.join(save_path, "runs", "activation_pca", model_alias)
    model_alias = os.path.basename(model_path)

    cfg = Config(model_alias=model_alias, model_path=model_path,
                 save_path=save_path, jailbreak_type="evil_confidant")
    # model_base = construct_model_base(cfg.model_path)
    save_path = artifact_path

    evaluate_completions_and_save_results_for_dataset(cfg, data_type,
                                                      cfg.jailbreak_eval_methodologies,
                                                      contrastive_type,
                                                      save_path)


if __name__ == "__main__":
    args = parse_arguments()
    data_type = ["jailbreakbench", "harmless"]

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 data_type=data_type, contrastive_type=tuple(args.contrastive_type),
                 )