import random
import json
import os
import argparse
from pipeline.SAE_config_cache_feature_activation import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.activation_pca import plot_contrastive_activation_pca, plot_contrastive_activation_intervention_pca
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.activation_pca import get_activations
from pipeline.submodules.activation_pca import generate_get_contrastive_activations_and_plot_pca
from dataset.load_dataset import load_dataset_split, load_dataset
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


def parse_arguments():

    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help="google/gemma-2-2b-it")
    parser.add_argument('--sae_release', type=str, required=False, default="gemma-scope-2b-pt-res")
    parser.add_argument('--sae_id', type=str, required=False, default="layer_20/width_16k/average_l0_71")
    parser.add_argument('--save_path', type=str, required=False, default=16)

    return parser.parse_args()


def list_flatten(nested_list):
    return [x for y in nested_list for x in y]


# A very handy function Neel wrote to get context around a feature activation
def make_token_df(model, tokens, len_prefix=5, len_suffix=3):
    str_tokens = [model.to_str_tokens(t) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    prompt = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            prompt.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        prompt=prompt,
        pos=pos,
        label=label,
    ))


def cache_feature_activations(cfg, model, sae, activation_store):

    # feature_list = torch.randint(0, sae.cfg.d_sae, (100,))
    feature_list = torch.arange(0,  sae.cfg.d_sae)
    examples_found = 0
    all_fired_tokens = []
    all_feature_acts = [] #
    all_reconstructions = []
    all_token_dfs = []

    total_batches = cfg.n_batches # 100
    batch_size_prompts = activation_store.store_batch_size_prompts
    batch_size_tokens = activation_store.context_size * batch_size_prompts
    pbar = tqdm(range(total_batches))
    for i in pbar:
        tokens = activation_store.get_batch_tokens()
        tokens_df = make_token_df(model, tokens)
        tokens_df["batch"] = i

        flat_tokens = tokens.flatten()

        _, cache = model.run_with_cache(tokens, stop_at_layer=sae.cfg.hook_layer + 1, names_filter=[sae.cfg.hook_name])
        sae_in = cache[sae.cfg.hook_name]
        feature_acts = sae.encode(sae_in).squeeze()

        feature_acts = feature_acts.flatten(0, 1)
        fired_mask = (feature_acts[:, feature_list]).sum(dim=-1) > 0
        fired_tokens = model.to_str_tokens(flat_tokens[fired_mask])
        # reconstruction = feature_acts[fired_mask][:, feature_list] @ sae.W_dec[feature_list]

        token_df = tokens_df.iloc[fired_mask.cpu().nonzero().flatten().numpy()]
        all_token_dfs.append(token_df)
        all_feature_acts.append(feature_acts[fired_mask][:, feature_list])
        all_fired_tokens.append(fired_tokens)
        # all_reconstructions.append(reconstruction)

        examples_found += len(fired_tokens)
        # update description
        pbar.set_description(f"Examples found: {examples_found}")

    # flatten the list of lists
    all_token_dfs = pd.concat(all_token_dfs)
    all_fired_tokens = list_flatten(all_fired_tokens)
    # all_reconstructions = torch.cat(all_reconstructions)
    all_feature_acts = torch.cat(all_feature_acts)

    return all_feature_acts, all_token_dfs


def run_pipeline(model_path,
                 sae_release,
                 sae_id,
                 save_path,
                 ):
    model_alias = os.path.basename(model_path)
    layer = sae_id.split('/')[0].split('_')[-1]
    width = sae_id.split('/')[1].split('_')[-1]
    l0 = sae_id.split('/')[-1].split('_')[-1]
    cfg = Config(model_alias=model_alias,
                 model_path=model_path,
                 sae_release=sae_release,
                 sae_id=sae_id,
                 save_path=save_path,
                 layer=layer,
                 width=width,
                 l0=l0
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

    # 2. instantiate an object to hold activations from a dataset
    # a convenient way to instantiate an activation store is to use the from_sae method
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        # fairly conservative parameters here so can use same for larger
        # models without running out of memory.
        store_batch_size_prompts=cfg.batch_size,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=32,
        context_size=128,
        device="cuda",
    )

    # 3. cache feature activation
    all_feature_acts, all_token_dfs = cache_feature_activations(cfg, model, sae, activation_store)

    # 4. store
    activation_store = {
        'all_feature_acts': all_feature_acts,
        'all_token_dfs': all_token_dfs
    }

    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, 'SAE_activation_cache')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + os.sep + f'feature_activation_store_layer_{layer}_width_{width}_l0_{l0}.pkl', "wb") as f:
        pickle.dump(activation_store, f)


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


    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 sae_release=args.sae_release, sae_id=args.sae_id,
                 )
