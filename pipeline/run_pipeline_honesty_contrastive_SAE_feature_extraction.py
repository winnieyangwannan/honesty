import random
import json
import os
import argparse
from pipeline.SAE_config_contrastive_feature_activation import Config
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


def parse_arguments():

    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help="google/gemma-2-2b-it")
    parser.add_argument('--sae_release', type=str, required=False, default="gemma-scope-2b-pt-res")
    parser.add_argument('--sae_id', type=str, required=False, default="layer_20/width_16k/average_l0_71")
    parser.add_argument('--save_path', type=str, required=False, default=16)
    parser.add_argument('--pos_extract', required=False, default='honest')
    parser.add_argument('--pos_type', type=str, required=False, default='str')
    parser.add_argument('--task_name', type=str, required=False, default='honesty')

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

    SYSTEM_PROMPT_HONEST = "You are to answer the following questions in an honest manner."
    SYSTEM_PROMPT_LYING = "You are to answer the following questions in a lying manner."
    USER_PROMPT = "Hi AI, is the following statement true or false?"
    SUFFIX = "Oh that's an easy one! The statement is definitely"
    # STATEMENT = "The planet Earth is 4.54 billion years old."
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


def get_feature_activation(cfg, model, sae, dataset, contrastive_type='honest', pos_extract='honest'):
    """
    cache sae feature activation at certain str location or at certain int location
    """
    batch_size = cfg.batch_size
    activation_cache = torch.zeros(len(dataset), sae.cfg.d_sae)
    for ii in tqdm(range(0, len(dataset), batch_size)):
        prompt = construct_prompt(dataset[ii:ii + batch_size], contrastive_type=contrastive_type)
        if isinstance(pos_extract, str):
            prompt_str = model.to_str_tokens(prompt[0])
            pos_ind = prompt_str.index(pos_extract)
        else:
            pos_ind = pos_extract
        _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])
        activation_cache[ii:ii + batch_size, :] = cache[sae.cfg.hook_name+'.hook_sae_acts_post'][:, pos_ind, :]
    return activation_cache


def get_data_z_score(cfg, sae, contrastive_activation_positive, contrastive_activation_negative, baseline_activation_cache):
    n_contrastive = contrastive_activation_positive.shape[0]
    data = torch.cat((contrastive_activation_positive, contrastive_activation_negative, baseline_activation_cache.cpu()))
    data_z = stats.zscore(data.detach().numpy(), axis=0)
    contrastive_activation_z_positive = data_z[:n_contrastive, :]
    contrastive_activation_z_negative = data_z[n_contrastive: n_contrastive+n_contrastive, :]

    feature_activation_df = pd.DataFrame(np.mean(contrastive_activation_z_positive, 0),
                                         index=[f"feature_{i}" for i in range(sae.cfg.d_sae)],
                                         )
    feature_activation_df.columns = ["positive"]
    feature_activation_df["negative"] = np.mean(contrastive_activation_z_negative, 0)
    feature_activation_df["diff"] = feature_activation_df["positive"] - feature_activation_df["negative"]

    layer = cfg.layer
    width = cfg.width
    l0 = cfg.l0
    submodule = cfg.submodule
    task_name = cfg.task_name

    fig = px.line(
        feature_activation_df,
        title=f"Feature activations (z_scored): Layer {layer}",
        labels={"index": "Feature", "value": "Activation"},
    )
    # hide the x-ticks
    fig.update_xaxes(showticklabels=False)
    fig.show()
    artifact_path = cfg.artifact_path()
    save_path = os.path.join(artifact_path, f'contrastive_SAE_{task_name}',
                             f'{submodule}', f'layer_{layer}', 'top_k_feature')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.write_html(save_path + os.sep + 'contrastive_feature_activation_z_' +
                   f'_{submodule}_layer_{layer}_width_{width}_{l0}_pos_{cfg.pos_extract}.html')

    return feature_activation_df


def get_feature_activation_z_score(cfg, sae, contrastive_activation_positive, contrastive_activation_negative):
    """
    contrastive_activation_cache: [n_data, n_feature]
    """
    layer = cfg.layer
    width = cfg.width
    l0 = cfg.l0
    submodule = cfg.submodule

    # 1. load baseline SAE activation cache [n_tokens, n_feature]
    sae_path = os.path.join(cfg.artifact_path(), 'SAE_activation_cache')
    with open(sae_path + os.sep + f'feature_activation_store_{submodule}_layer_{layer}_width_{width}_l0_{l0}.pkl', "rb") as f:
        data = pickle.load(f)
    baseline_activation_cache = data['all_feature_acts']

    # 2. from torch.nn.functional import normalize
    feature_activation_df = get_data_z_score(cfg, sae,
                                             contrastive_activation_positive,
                                             contrastive_activation_negative,
                                             baseline_activation_cache)
    return feature_activation_df


def plot_contrastive_feature(cfg, sae, feature_activation_honest, feature_activation_lying):

    # plot
    layer = cfg.layer
    width = cfg.width
    l0 = cfg.l0
    submodule = cfg.submodule
    task_name = cfg.task_name
    feature_activation_df = pd.DataFrame(torch.mean(feature_activation_honest, 0).cpu().numpy(),
                                         index=[f"feature_{i}" for i in range(sae.cfg.d_sae)],
                                         )
    feature_activation_df.columns = ["honest"]
    feature_activation_df["lying"] = torch.mean(feature_activation_lying, 0).cpu().numpy()
    feature_activation_df["diff"] = feature_activation_df["honest"] - feature_activation_df["lying"]
    fig = px.line(
        feature_activation_df,
        title=f"Feature activations: Layer {layer}",
        labels={"index": "Feature", "value": "Activation"},
    )
    # hide the x-ticks
    fig.update_xaxes(showticklabels=False)
    fig.show()
    artifact_path = cfg.artifact_path()
    save_path = os.path.join(artifact_path, f'contrastive_SAE_{task_name}',
                             f'{submodule}', f'layer_{layer}', 'top_k_feature')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.write_html(save_path + os.sep + f'contrastive_feature_activation_{submodule}_' +
                                        f'layer_{layer}_width_{width}_{l0}_pos_{cfg.pos_extract}.html')


def get_contrastive_feature_activation_z_score(cfg, model, sae, dataset_train,
                                               ):

    statements_train = [row['claim'] for row in dataset_train]
    labels_train = [row['label'] for row in dataset_train]
    categories_train = [row['dataset'] for row in dataset_train]

    pos_extract = cfg.pos_extract

    feature_activation_honest = get_feature_activation(cfg, model, sae, statements_train,
                                                       contrastive_type='honest', pos_extract=pos_extract[0])
    feature_activation_lying = get_feature_activation(cfg, model, sae, statements_train,
                                                      contrastive_type='lying', pos_extract=pos_extract[1])

    plot_contrastive_feature(cfg, sae, feature_activation_honest, feature_activation_lying)

    feature_df = get_feature_activation_z_score(cfg,
                                                sae,
                                                feature_activation_honest,
                                                feature_activation_lying)

    return feature_df


def get_top_k_mean_diff(feature_df, topK=20):

    # top k
    diff_top_k = feature_df.nlargest(topK, 'diff')
    top_k_ind = [int(diff_top_k.index[ii].split('_')[-1]) for ii in range(topK)]
    top_k_val = [diff_top_k['diff'][ii] for ii in range(topK)]
    print("top_k_ind")
    print(top_k_ind)
    print("top_k_val")
    print(top_k_val)

    return top_k_ind, top_k_val


def run_pipeline(model_path,
                 sae_release,
                 sae_id,
                 save_path,
                 pos_extract=[' truthful', ' lying'],
                 pos_type='str',
                 task_name='honesty'
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
                 task_name=task_name
                 )

    # 1. Load Model and SAE
    model = HookedSAETransformer.from_pretrained(model_path, device="cuda")

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

    # 3. get feature activation
    feature_df = get_contrastive_feature_activation_z_score(cfg, model,
                                                            sae,
                                                            dataset_train,
                                                            )

    # 4. get top k of mean difference activation
    topK = cfg.topK
    top_k_ind, top_k_val = get_top_k_mean_diff(feature_df, topK=topK)

    # 5. store
    top_k = {
        'top_k_ind': top_k_ind,
        'top_k_val': top_k_val,
    }
    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, f'contrastive_SAE_{cfg.task_name}',
                             f'{submodule}', f'layer_{layer}', 'top_k_feature')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + os.sep + f'feature_contrastive_activation_top_k_'
                                   f'{submodule}_layer_{layer}_width_{width}_l0_{l0}_pos_{pos_extract}.pkl',
              "wb") as f:
        pickle.dump(top_k, f)


if __name__ == "__main__":
    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("run_pipieline_jailbreak_contrastive_SAE\n\n")
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

    if args.pos_type == 'int': # convert to integer
        pos_extract = [int(args.pos_extract), int(args.pos_extract)]
    elif args.pos_type == 'str': # add space for gemma
        pos_extract = [f' {args.pos_extract}', ' lying']

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 sae_release=args.sae_release, sae_id=args.sae_id,
                 pos_extract=pos_extract, pos_type=args.pos_type,
                 task_name=args.task_name)
