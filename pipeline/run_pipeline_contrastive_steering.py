import random
import json
import os
import argparse
from pipeline.configs.config_contrastive_steering import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.activation_pca import plot_contrastive_activation_pca, plot_contrastive_activation_intervention_pca
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.activation_pca import get_activations
from pipeline.submodules.activation_pca import generate_get_contrastive_activations_and_plot_pca
from dataset.load_dataset import load_dataset_split
from datasets import load_dataset
import numpy as np
import sae_lens
import transformer_lens
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import torch
import pickle
from sae_lens import ActivationsStore
from scipy import stats
from functools import partial


def parse_arguments():

    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help="google/gemma-2-2b-it")
    parser.add_argument('--source_layer', type=int, required=False, default=0)
    parser.add_argument('--target_layer', type=int, required=False, default=0)
    parser.add_argument('--save_path', type=str, required=False, default=16)
    parser.add_argument('--hook_name', type=str, required=False, default='resid_post')
    parser.add_argument('--intervention', type=str, required=False, default='positive_addition')
    parser.add_argument('--task_name', type=str, required=False, default='honesty')
    parser.add_argument('--jailbreak', type=str, required=False, default='evil_confidant')

    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    category = 'facts'
    model_name = cfg.model_alias

    dataset_all = load_dataset("notrichardren/azaria-mitchell-diff-filtered-2")
    dataset = [row for row in dataset_all[f"{category}"]]
    dataset_train = random.sample(dataset, cfg.n_train)
    dataset_test = random.sample(dataset, cfg.n_test)

    return dataset_train, dataset_test


def construct_prompt(statements, contrastive_type="honest"):

    SYSTEM_PROMPT_HONEST = "You are to answer the following questions in a truthful manner."
    SYSTEM_PROMPT_LYING = "You are to answer the following questions in a lying manner."
    USER_PROMPT = "Hi AI, is the following statement true or false?"
    SUFFIX = "Oh that's an easy one! The statement is definitely"
    GEMMA_CHAT_TEMPLATE_STATEMENT = """<start_of_turn>user
    {system_prompt}\n\n{user_prompt}\n\nStatement: {statement}\n\n<end_of_turn>
    <start_of_turn>model
    """

    if contrastive_type == "honest":
        prompt = [GEMMA_CHAT_TEMPLATE_STATEMENT.format(system_prompt=SYSTEM_PROMPT_HONEST,
                                                       user_prompt=USER_PROMPT,
                                                       statement=statement) + SUFFIX for statement in statements]
    elif contrastive_type == "lying":
        prompt = [GEMMA_CHAT_TEMPLATE_STATEMENT.format(system_prompt=SYSTEM_PROMPT_LYING,
                                                       user_prompt=USER_PROMPT,
                                                       statement=statement) + SUFFIX for statement in statements]
    return prompt


def steering(activations, hook, cfg=None, steering_strength=1.0, steering_vector=None):
    # print(steering_vector.shape) # [batch, n_tokens, n_head, d_head ]
    if cfg.intervention == 'positive_addition':
        return activations + steering_strength * steering_vector


def generate_with_steering(cfg, model, steering_vector,
                           statements, labels,
                           contrastive_type='honest'):

    max_new_tokens = cfg.max_new_tokens
    batch_size = cfg.batch_size
    task_name = cfg.task_name

    source_layer = cfg.source_layer
    target_layer = cfg.target_layer
    hook_name = cfg.hook_name
    intervention = cfg.intervention
    steering_strength = cfg.steering_strength

    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, intervention, 'completions')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    completions = []
    for ii in tqdm(range(0, len(statements), batch_size)):

        # 1. prompt to input
        prompt = construct_prompt(statements[ii:ii + batch_size], contrastive_type=contrastive_type)
        input_ids = model.to_tokens(prompt, prepend_bos=model.cfg.default_prepend_bos)

        # 3. Steering hook
        steering_hook = partial(
                                steering,
                                cfg=cfg,
                                steering_vector=steering_vector,
                                steering_strength=steering_strength,
                                )

        # 4. Generation with hook
        with model.hooks(fwd_hooks=[(f'blocks.{source_layer}.hook_{hook_name}', steering_hook)]):
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0,
                # top_p=0.9,
                # stop_at_eos = False if device == "mps" else True,
                stop_at_eos=False,
                prepend_bos=model.cfg.default_prepend_bos,
            )

        # 5. Get generation output (one batch)
        generation_toks = model.tokenizer.batch_decode(output[:, input_ids.shape[-1]:],
                                                       skip_special_tokens=True)
        for generation_idx, generation in enumerate(generation_toks):
            completions.append({
                'prompt': statements[ii + generation_idx],
                'response': generation.strip(),  # tokenizer.decode(generation, skip_special_tokens=True).strip(),
                'label': labels[ii + generation_idx],
                'ID': ii + generation_idx
            })

    # 6. Store all generation results (all batches)
    with open(
            save_path + os.sep + f'contrastive_steering_generation_{intervention}_'
                                 f'layer_s_{source_layer}_layer_t_{target_layer}_{contrastive_type}.json',
            "w") as f:
        json.dump(completions, f, indent=4)


def run_with_cache(cfg, model, statements, contrastive_type='honest'):

    batch_size = cfg.batch_size
    source_layer = cfg.source_layer
    hook_name = cfg.hook_name

    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, 'activation_cache')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    activations = []
    for ii in tqdm(range(0, len(statements), batch_size)):

        # 1. prompt to input
        prompt = construct_prompt(statements[ii:ii + batch_size], contrastive_type=contrastive_type)
        input_ids = model.to_tokens(prompt, prepend_bos=model.cfg.default_prepend_bos)

        # 2. run with cache
        _, cache = model.run_with_cache(input_ids,
                                        stop_at_layer=source_layer+1,
                                        names_filter=[f'blocks.{source_layer}.hook_{hook_name}'])
        activations.append(cache[f'blocks.{source_layer}.hook_{hook_name}'][:, -1, :])

    # 3. save activation
    activations_all = torch.cat(activations)

    return activations_all


def construct_steering_vector(cfg, activations_positive, activations_negative):

     # 1. get mean activation
     mean_activation_positive = activations_positive[:cfg.n_train].mean(dim=0)  # only use harmful data
     mean_activation_negative = activations_negative[:cfg.n_train].mean(dim=0)

     # 2. get mean difference
     if cfg.intervention == 'positive_addition':
        mean_diff = mean_activation_positive - mean_activation_negative

     return mean_diff


def get_contrastive_steering_vector(cfg, model, dataset):

    # 0. Get data
    statements = [row['claim'] for row in dataset]
    labels = [row['label'] for row in dataset]

    contrastive_type = cfg.contrastive_type

    # 1. generate with cache
    # positive
    activations_positive = run_with_cache(cfg, model,
                                          statements,
                                          contrastive_type=contrastive_type[0])
    # negative
    activations_negative = run_with_cache(cfg, model,
                                          statements,
                                          contrastive_type=contrastive_type[1])

    # 2. get steering vector
    mean_diff = construct_steering_vector(cfg, activations_positive, activations_negative)

    return mean_diff


def contrastive_generation_with_steering(cfg, model, dataset, steering_vector):

    statements = [row['claim'] for row in dataset]
    labels = [row['label'] for row in dataset]

    contrastive_type = cfg.contrastive_type

    # positive
    generate_with_steering(cfg, model, steering_vector,
                           statements, labels,
                           contrastive_type[0])
    # negative
    generate_with_steering(cfg, model, steering_vector,
                           statements, labels,
                           contrastive_type[1])


def run_pipeline(model_path='google/gemma-2-2b-it',
                 source_layer=0,
                 target_layer=0,
                 hook_name='resid_pre',
                 task_name='honesty',
                 contrastive_type=['honesty', 'lying'],
                 save_path='D:\Data\honesty',
                 intervention='positive_addition'
                 ):

    model_alias = os.path.basename(model_path)

    cfg = Config(model_alias=model_alias,
                 model_path=model_path,
                 source_layer=source_layer,
                 target_layer=target_layer,
                 intervention=intervention,
                 hook_name=hook_name,
                 save_path=save_path,
                 task_name=task_name,
                 contrastive_type=contrastive_type
                 )

    # 1. Load Model
    model = HookedSAETransformer.from_pretrained(model_path, device="cuda")
    model.tokenizer.padding_side = 'left'

    # 2. Load data
    dataset_train, dataset_test = load_and_sample_datasets(cfg)

    # 3. Extract steering vector
    steering_vector = get_contrastive_steering_vector(cfg, model, dataset_train)

    # 4. generate with steer
    contrastive_generation_with_steering(cfg, model, dataset_test, steering_vector)


if __name__ == "__main__":
    args = parse_arguments()

    if args.task_name == 'honesty':
        contrastive_type = ['honest', 'lying']
    elif args.task_name == 'jailbreak':
        contrastive_type = ['HHH', args.jailbreak]

    print("run_pipieline_contrastive_steering\n\n")
    print('task_name')
    print(args.task_name)
    print("model_path")
    print(args.model_path)
    print("save_path")
    print(args.save_path)
    print("source_layer")
    print(args.source_layer)
    print("target_layer")
    print(args.target_layer)
    print("hook_name")
    print(args.hook_name)
    print("task_name")
    print(args.task_name)
    print("contrastive_type")
    print(contrastive_type)
    print("intervention")
    print(args.intervention)

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 source_layer=args.source_layer, target_layer=args.target_layer,
                 hook_name=args.hook_name, intervention=args.intervention,
                 task_name=args.task_name, contrastive_type=contrastive_type)
