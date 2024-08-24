import random
import json
import os
import argparse
from pipeline.SAE_config_contrastive_feature_steering import Config
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
    parser.add_argument('--sae_release', type=str, required=False, default="gemma-scope-2b-pt-res")
    parser.add_argument('--sae_id', type=str, required=False, default="layer_20/width_16k/average_l0_71")
    parser.add_argument('--save_path', type=str, required=False, default=16)
    parser.add_argument('--pos_extract', required=False, default='truthful')
    parser.add_argument('--pos_type', type=str, required=False, default='str')
    parser.add_argument('--task_name', type=str, required=False, default='str')
    parser.add_argument('--jailbreak', type=str, required=False, default='str')

    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    category = cfg.data_category
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


def get_topk_contrastive_feature(cfg):
    topK = cfg.topK
    task_name = cfg.task_name
    submodule = cfg.submodule
    layer = cfg.layer
    width = cfg.width
    l0 = cfg.l0
    pos_extract = cfg.pos_extract
    artifact_dir = cfg.artifact_path()

    save_path = os.path.join(artifact_dir, f'contrastive_SAE_{task_name}',
                             f'{submodule}', f'layer_{layer}', 'top_k_feature')
    with open(save_path + os.sep + f'feature_contrastive_activation_top_k_{submodule}_'
              f'layer_{layer}_width_{width}_l0_{l0}_pos_{pos_extract}.pkl',
              "rb") as f:
        data = pickle.load(f)
    top_k_ind = data['top_k_ind'][:topK]

    return top_k_ind


def get_feature_activation_distribution_stats(feature_activations):

    # 1. get mean
    mean_activations = np.mean(feature_activations.detach().cpu().numpy(), axis=0)

    # 2. get 95 percentile
    perc_95 = np.percentile(feature_activations.detach().cpu().numpy(), 95, axis=0)

    # 3. get 99 percentile
    perc_99 = np.percentile(feature_activations.detach().cpu().numpy(), 99, axis=0)

    # 4. maximum
    max_activations = np.max(feature_activations.detach().cpu().numpy(), axis=0)

    activation_stats = {
        'mean': mean_activations,
        '95': perc_95,
        '99': perc_99,
        'max': max_activations
    }
    return activation_stats


def get_baseline_contrastive_feature_activation(cfg, feature_list):

    # 1. load baseline SAE activation cache [n_tokens, n_feature]
    layer = cfg.layer
    width = cfg.width
    l0 = cfg.l0
    submodule = cfg.submodule

    sae_path = os.path.join(cfg.artifact_path(), 'SAE_activation_cache')
    with open(sae_path + os.sep + f'feature_activation_store_{submodule}_layer_{layer}_width_{width}_l0_{l0}.pkl', "rb") as f:
        data = pickle.load(f)
    baseline_activation_cache = data['all_feature_acts']
    feature_activations = baseline_activation_cache[:, feature_list]

    # 2. get statistics of feature activation distrubution
    feature_stats = get_feature_activation_distribution_stats(feature_activations)

    return feature_stats


def steering(activations, hook, steering_strength=1.0, steering_vector=None, max_act=1.0):
    # Note if the feature fires anyway, we'd be adding to that here.
    # print(activations.shape) # [batch, n_tokens, n_head, d_head ]
    # steering_vector = torch.reshape(steering_vector,(model.cfg.n_heads, model.cfg.d_head)) # for attn
    # print("steering_vector.shape") # [batch, n_tokens, n_head, d_head ]

    # print(steering_vector.shape) # [batch, n_tokens, n_head, d_head ]
    return activations + max_act * steering_strength * steering_vector


def generate_without_steering(cfg, model, sae,
                           statements, labels,
                           feature_id,
                           max_act, steering_strength,
                           contrastive_type):

    max_new_tokens = cfg.max_new_tokens
    batch_size = cfg.batch_size
    task_name = cfg.task_name

    layer = cfg.layer
    width = cfg.width
    l0 = cfg.l0
    submodule = cfg.submodule

    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, f'contrastive_SAE_{task_name}',
                             f'{submodule}', f'layer_{layer}', 'completion')

    completions = []
    for ii in tqdm(range(0, len(statements), batch_size)):

        # 1. prompt to input
        prompt = construct_prompt(statements[ii:ii + batch_size], contrastive_type=contrastive_type)
        input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)

        # 2. Steering vector
        # extracted form the decoder weight
        steering_vector = sae.W_dec[feature_id].to(model.cfg.device)

        # 4. Generation with hook
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0,
            # top_p=0.9,
            # stop_at_eos = False if device == "mps" else True,
            stop_at_eos=False,
            prepend_bos=sae.cfg.prepend_bos)

        # 5. Get generation output (one batch)
        generation_toks = output[:, input_ids.shape[-1]:]
        for generation_idx, generation in enumerate(generation_toks):
            completions.append({
                'prompt': statements[ii + generation_idx],
                'response': model.tokenizer.decode(generation, skip_special_tokens=True).strip(),
                'label': labels[ii + generation_idx],
                'ID': ii + generation_idx
            })

    # 6. Store all generation results (all batches)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(
            save_path + os.sep + f'SAE_steering_generation_{submodule}_'
                                 f'layer_{layer}_width_{width}_l0_{l0}_pos_{pos_extract}_'
                                 f'id_{feature_id}_x{steering_strength}_{contrastive_type}.json',
            "w") as f:
        json.dump(completions, f, indent=4)


def generate_with_steering(cfg, model, sae,
                           statements, labels,
                           feature_id,
                           max_act, steering_strength,
                           contrastive_type):

    max_new_tokens = cfg.max_new_tokens
    batch_size = cfg.batch_size
    task_name = cfg.task_name

    layer = cfg.layer
    width = cfg.width
    l0 = cfg.l0
    submodule = cfg.submodule

    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, f'contrastive_SAE_{task_name}',
                             f'{submodule}', f'layer_{layer}', 'completion')

    completions = []
    for ii in tqdm(range(0, len(statements), batch_size)):

        # 1. prompt to input
        prompt = construct_prompt(statements[ii:ii + batch_size], contrastive_type=contrastive_type)
        input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)

        # 2. Steering vector
        # extracted form the decoder weight
        steering_vector = sae.W_dec[feature_id].to(model.cfg.device)

        # 3. Steering hok
        steering_hook = partial(
            steering,
            steering_vector=steering_vector,
            steering_strength=steering_strength,
            max_act=max_act
        )

        # 4. Generation with hook
        with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0,
                # top_p=0.9,
                # stop_at_eos = False if device == "mps" else True,
                stop_at_eos=False,
                prepend_bos=sae.cfg.prepend_bos,
            )

        # 5. Get generation output (one batch)
        generation_toks = model.tokenizer.batch_decode(output[:, input_ids.shape[-1]:],  skip_special_tokens=True)
        for generation_idx, generation in enumerate(generation_toks):
            completions.append({
                'prompt': statements[ii + generation_idx],
                'response': generation.strip(),  # tokenizer.decode(generation, skip_special_tokens=True).strip(),
                'label': labels[ii + generation_idx],
                'ID': ii + generation_idx
            })

    # 6. Store all generation results (all batches)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(
            save_path + os.sep + f'SAE_steering_generation_{submodule}_'
                                 f'layer_{layer}_width_{width}_l0_{l0}_pos_{pos_extract}_'
                                 f'id_{feature_id}_x{steering_strength}_{contrastive_type}.json',
            "w") as f:
        json.dump(completions, f, indent=4)


def generate_with_steering_features(cfg, model, sae, dataset, steering_feature, max_act):

    statements = [row['claim'] for row in dataset]
    labels = [row['label'] for row in dataset]
    categories = [row['dataset'] for row in dataset]
    steering_strengths = [-8, -6, -4, -2, 2, 4, 6, 8, 10]
    for steering_strength in steering_strengths:
        print(f"steering_strengths: {steering_strengths}")
        for ff, feature_id in enumerate(steering_feature):
            print(f"feature_id: {feature_id}")

            # positive
            generate_with_steering(cfg, model, sae,
                                   statements, labels,
                                   feature_id,
                                   max_act[ff], steering_strength,
                                   contrastive_type[0])
            # negative
            generate_with_steering(cfg, model, sae,
                                   statements, labels,
                                   feature_id,
                                   max_act[ff], steering_strength,
                                   contrastive_type[1])


def run_pipeline(model_path,
                 sae_release,
                 sae_id,
                 save_path,
                 pos_extract=[' truthful', ' lying'],
                 pos_type='str',
                 task_name='honesty',
                 contrastive_type=['honesty', 'lying']
                 ):

    model_alias = os.path.basename(model_path)
    layer = sae_id.split('/')[0].split('_')[-1]
    width = sae_id.split('/')[1].split('_')[-1]
    l0 = sae_id.split('/')[-1].split('_')[-1]
    submodule = sae_release.split('-')[-1]

    cfg = Config(model_alias=model_alias,
                 model_path=model_path,
                 sae_release=sae_release,
                 sae_id=sae_id,
                 save_path=save_path,
                 submodule=submodule,
                 layer=layer,
                 width=width,
                 l0=l0,
                 pos_extract=pos_extract,
                 pos_type=pos_type,
                 task_name=task_name,
                 contrastive_type=contrastive_type
                 )

    # 1. Load Model and SAE
    model = HookedSAETransformer.from_pretrained(model_path, device="cuda")
    model.tokenizer.padding_side = 'left'

    # the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
    # Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
    # We also return the feature sparsities which are stored in HF for convenience.
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,  # <- Release name
        sae_id=sae_id,  # <- SAE id (not always a hook point!)
        device="cuda"
    )

    # 2. Load data
    dataset_train, dataset_test = load_and_sample_datasets(cfg)

    # 3, get the list of feature to steer
    top_k_ind = get_topk_contrastive_feature(cfg)

    # 3. get baseline feature activation ( mean , z, max)
    feature_stats = get_baseline_contrastive_feature_activation(cfg, top_k_ind)
    max_act = feature_stats[cfg.steering_type]

    # 4. generate with steer
    generate_with_steering_features(cfg, model, sae, dataset_train, top_k_ind, max_act)


if __name__ == "__main__":
    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("run_pipieline_jailbreak_contrastive_SAE\n\n")
    print('task_name')
    print(args.task_name)
    print("model_path")
    print(args.model_path)
    print("save_path")
    print(args.save_path)
    print("sae_release")
    print(args.sae_release)
    print("sae_id")
    print(args.sae_id)
    print("pos_extract")
    print(args.pos_extract)
    print("pos_type")
    print(args.pos_type)

    if args.pos_type == 'int':  # convert to integer
        pos_extract = [int(args.pos_extract), int(args.pos_extract)]
    elif args.pos_type == 'str':  # add space for gemma
        pos_extract = [' truthful', ' lying']

    if args.task_name == 'honesty':
        contrastive_type = ['honest', 'lying']
    elif args.task_name == 'jailbreak':
        contrastive_type = ['HHH', args.jailbreak]

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 sae_release=args.sae_release, sae_id=args.sae_id,
                 pos_extract=pos_extract, pos_type=args.pos_type,
                 task_name=args.task_name, contrastive_type=contrastive_type)
