import random
import json
import os
import argparse
from pipeline.configs.config_SAE_steering import Config
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
from scipy import stats
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
    parser.add_argument('--pos_extract', required=False, default='honest')
    parser.add_argument('--pos_type', type=str, required=False, default='str')
    parser.add_argument('--task_name', type=str, required=False, default='honesty')
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


def construct_steering_prompt(contrastive_label="honest"):

    SYSTEM_PROMPT_HONEST = "You are to answer the following questions in an honest manner."
    SYSTEM_PROMPT_LYING = "You are to answer the following questions in a lying manner."

    GEMMA_CHAT_TEMPLATE_STATEMENT = """<start_of_turn>user
    {system_prompt}<end_of_turn>
    <start_of_turn>model
    """

    if contrastive_label == "honest":
        prompt = GEMMA_CHAT_TEMPLATE_STATEMENT.format(system_prompt=SYSTEM_PROMPT_HONEST)
    elif contrastive_label == "lying":
        prompt = GEMMA_CHAT_TEMPLATE_STATEMENT.format(system_prompt=SYSTEM_PROMPT_LYING)
    return prompt


def construct_prompt(statements, contrastive_label="honest"):

    SYSTEM_PROMPT_HONEST = "You are to answer the following questions in an honest manner."
    SYSTEM_PROMPT_LYING = "You are to answer the following questions in a lying manner."
    USER_PROMPT = "Hi AI, is the following statement true or false?"
    SUFFIX = "Oh that's an easy one! The statement is definitely"
    GEMMA_CHAT_TEMPLATE_STATEMENT = """<start_of_turn>user
    {system_prompt}\n\n{user_prompt}\n\nStatement: {statement}\n\n<end_of_turn>
    <start_of_turn>model
    """

    if contrastive_label == "honest":
        prompt = [GEMMA_CHAT_TEMPLATE_STATEMENT.format(system_prompt=SYSTEM_PROMPT_HONEST,
                                                       user_prompt=USER_PROMPT,
                                                       statement=statement) + SUFFIX for statement in statements]
    elif contrastive_label == "lying":
        prompt = [GEMMA_CHAT_TEMPLATE_STATEMENT.format(system_prompt=SYSTEM_PROMPT_LYING,
                                                       user_prompt=USER_PROMPT,
                                                       statement=statement) + SUFFIX for statement in statements]
    return prompt


def plot_token_feature_acts(cfg, activation_cache, inds, vals, contrastive_label):
    inds = inds.detach().cpu().numpy()
    vals = vals.detach().cpu().numpy()
    n_feature = len(activation_cache)
    layer = cfg.layer

    activation = activation_cache.detach().cpu().numpy()
    activation_z = stats.zscore(activation)
    # plot activation distribution for this token
    # feature_activation_z = {
    #     'Activations': activation,
    #     'Activation (zscored)': activation_z
    # }
    feature_activation_df_z = pd.DataFrame(activation_z,
                                         index=[f"feature_{i}" for i in range(n_feature)],
                                         )    # plot activation for all features for this token
    feature_activation_df = pd.DataFrame(activation,
                                         index=[f"feature_{i}" for i in range(n_feature)],
                                         )
    # 1.token feature activation histogram
    fig = px.histogram(
        feature_activation_df_z,
        log_y=True,
        nbins=50,
        title=f"Histogram of feature activations:  Layer {layer} at '{contrastive_label}' token",
        width=800, )
    fig.add_vline(x=activation_z[inds[0]],
                  label=dict(
                      text=f"feature: {inds[0]}",
                      textposition="top left",
                      font=dict(size=20, color="green"),
                  ),
                  line_width=3, line_dash="dash", line_color="green")
    fig.add_vline(x=activation_z[inds[2]],
                  label=dict(
                      text=f"feature: {inds[2]}",
                      textposition="top left",
                      font=dict(size=20, color="red"),
                  ),
                  line_width=3, line_dash="dash", line_color="red")
    fig.add_vline(x=np.percentile(activation, 99),
                  label=dict(
                      text=f"99 percentile",
                      textposition="top left",
                      font=dict(size=20, color="black"),
                  ),
                  line_width=3, line_dash="dash", line_color="black")
    fig.show()
    fig.update_layout(
        font=dict(size=20),
        width=1000, height=500,
        xaxis_title='Activations (z-scored)',
        yaxis_title='Count (log)'
    )
    pio.write_image(fig, cfg.data_save_path + os.sep + f'token_feature_activation_histogram_{contrastive_label}.png', scale=6, width=1080, height=1080)
    fig.write_html(cfg.data_save_path + os.sep + f'token_feature_activation_histogram_{contrastive_label}.html')

    # 2. Feature activation at token
    fig = px.line(
        feature_activation_df,
        title=f"Feature activations: Layer {layer}",
        labels={"index": "Feature", "value": "Activation"},
    )
    # hide the x-ticks
    fig.update_xaxes(showticklabels=False)
    fig.show()
    fig.add_annotation(
        x=inds[0],
        y=vals[0],
        text=f"feature: {inds[0]}",
        showarrow=True,
        xanchor="right",
        font=dict(
            color="green",
            size=20
        ),
    )
    fig.add_annotation(
        x=inds[2],
        y=vals[2],
        text=f"feature: {inds[2]}",
        showarrow=True,
        xanchor="right",
        font=dict(
            color="red",
            size=20
        ),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        font=dict(size=20)
    )
    fig.show()
    pio.write_image(fig, cfg.data_save_path + os.sep + f'token_feature_activation_{contrastive_label}.png',
                    width=1080, height=1080)
    fig.write_html(cfg.data_save_path + os.sep + f'token_feature_activation_{contrastive_label}.html')


def top_k_sae_activation(cfg, model, sae, contrastive_label=' honest', pos_extract=' honest'):
    # 1. Construct steering prompt
    prompt_positive = construct_steering_prompt(contrastive_label=contrastive_label)

    # 2. Feature extraction position
    if isinstance(pos_extract, str):
        prompt_str = model.to_str_tokens(prompt_positive)
        pos_ind = prompt_str.index(pos_extract)
    else:
        pos_ind = pos_extract

    # 3. Extract activation
    _, cache = model.run_with_cache_with_saes(prompt_positive, saes=[sae])
    activation_cache = cache[sae.cfg.hook_name + '.hook_sae_acts_post'][0, pos_ind, :]

    # 4. Get top k
    vals, inds = torch.topk(activation_cache.cpu(), cfg.topK)

    # 5. plot activation
    plot_token_feature_acts(cfg, activation_cache, inds, vals, contrastive_label)

    return inds


def get_topk_active_feature(cfg, model, sae):
    topK = cfg.topK
    task_name = cfg.task_name
    submodule = cfg.submodule
    layer = cfg.layer
    width = cfg.width
    l0 = cfg.l0
    pos_extract = cfg.pos_extract
    artifact_dir = cfg.artifact_path()
    contrastive_label  = cfg.contrastive_label

    topK_positive = top_k_sae_activation(cfg, model, sae, contrastive_label=contrastive_label[0], pos_extract=pos_extract[0])
    topK_negative = top_k_sae_activation(cfg, model, sae, contrastive_label=contrastive_label[1], pos_extract=pos_extract[1])

    top_k_ind = torch.stack((topK_positive, topK_negative))

    # save
    save_path = os.path.join(artifact_dir, f'topK_SAE_{task_name}',
                             f'{submodule}', f'layer_{layer}', 'top_k_feature')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + os.sep + f'feature_activation_top_k_{submodule}_'
              f'layer_{layer}_width_{width}_l0_{l0}_pos_{pos_extract}.pkl',
              "wb") as f:
        pickle.dump(top_k_ind, f)

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

    feature_activations_positive = baseline_activation_cache[:, feature_list[0]]
    feature_activations_negative = baseline_activation_cache[:, feature_list[1]]

    # 2. get statistics of feature activation distrubution
    feature_stats_positive = get_feature_activation_distribution_stats(feature_activations_positive)
    feature_stats_negative = get_feature_activation_distribution_stats(feature_activations_negative)

    # 3. get top activation examples
    all_token_dfs = baseline_activation_cache['all_token_dfs']
    max_activation_example_positive = all_token_dfs.iloc[feature_list[0]]
    max_activation_example_positive = all_token_dfs.iloc[feature_list[1]]

    return feature_stats_positive, feature_stats_negative


def steering(activations, hook, steering_strength=1.0, steering_vector=None, max_act=1.0):
    # print(steering_vector.shape) # [d_model]
    # print(activations.shape) # [batch, len_prompt, d_model] $ first pass on the prompt
    # print(activations.shape) # [batch, 1, d_model] # then pass on the generated answer one by one
    n_batch = activations.shape[0]
    n_pos = activations.shape[1]
    activation_add = max_act * steering_strength * steering_vector

    return activations + activation_add.repeat(n_batch, n_pos, 1)


def generate_with_steering(cfg, model, sae,
                           statements, labels,
                           feature_id,
                           max_act, steering_strength,
                           contrastive_label,
                           pos_extract):

    max_new_tokens = cfg.max_new_tokens
    batch_size = cfg.batch_size
    task_name = cfg.task_name

    layer = cfg.layer
    width = cfg.width
    l0 = cfg.l0
    submodule = cfg.submodule

    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, f'topK_SAE_{task_name}',
                             f'{submodule}', f'layer_{layer}', f'feature_{pos_extract}', 'completion')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    completions = []
    for ii in tqdm(range(0, len(statements), batch_size)):

        # 1. prompt to input
        prompt = construct_prompt(statements[ii:ii + batch_size], contrastive_label=contrastive_label)
        input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)

        # 2. Steering vector
        # extracted form the decoder weight
        steering_vector = sae.W_dec[feature_id].to(model.cfg.device)

        # 3. Steering hook
        steering_hook = partial(
            steering,
            steering_vector=steering_vector,
            steering_strength=steering_strength,
            max_act=max_act
        )

        # 4. Generation with hook
        model.reset_hooks()
        with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.1,
                freq_penalty=1.0,
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

    with open(
            save_path + os.sep + f'SAE_steering_generation_{submodule}_'
                                 f'layer_{layer}_width_{width}_l0_{l0}_feature_{pos_extract}_'
                                 f'id_{feature_id}_x{steering_strength}_persona_{contrastive_label}.json',
            "w") as f:
        json.dump(completions, f, indent=4)


def generate_with_steering_features(cfg, model, sae, dataset, steering_feature, max_act, pos_extract):

    statements = [row['claim'] for row in dataset]
    labels = [row['label'] for row in dataset]
    categories = [row['dataset'] for row in dataset]
    steering_strengths = [2, 4, 6, 8, 10]
    for steering_strength in steering_strengths:
        print(f"steering_strengths: {steering_strengths}")
        for ff, feature_id in enumerate(steering_feature):
            print(f"feature_id: {feature_id}")

            # positive persona
            generate_with_steering(cfg, model, sae,
                                   statements, labels,
                                   feature_id,
                                   max_act[ff], steering_strength,
                                   contrastive_label[0], pos_extract)
            # negative persona
            generate_with_steering(cfg, model, sae,
                                   statements, labels,
                                   feature_id,
                                   max_act[ff], steering_strength,
                                   contrastive_label[1], pos_extract)


def run_pipeline(model_path,
                 sae_release,
                 sae_id,
                 save_path,
                 pos_extract=[' honest', ' lying'],
                 pos_type='str',
                 task_name='honesty',
                 contrastive_label=['honesty', 'lying']
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
                 contrastive_label=contrastive_label
                 )
    data_save_path = os.path.join(cfg.artifact_path(), f'topK_SAE_{task_name}',
                                  f'{submodule}', f'layer_{layer}', 'top_k_feature')
    cfg.data_save_path = data_save_path
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    # 1. Load Model and SAE
    model = HookedSAETransformer.from_pretrained_no_processing(model_path, device="cuda",
                                                             dtype=torch.bfloat16)
    model.tokenizer.padding_side = 'left'
    print("free(Gb):", torch.cuda.mem_get_info()[0] / 1000000000, "total(Gb):",
          torch.cuda.mem_get_info()[1] / 1000000000)

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
    top_k_ind = get_topk_active_feature(cfg, model, sae)

    # 3. get baseline feature activation ( mean , z, max)
    feature_stats_positive, feature_stats_negative = get_baseline_contrastive_feature_activation(cfg, top_k_ind)

    # 4. get feature top activations
    max_act_positive = feature_stats_positive[cfg.steering_type]
    max_act_negative = feature_stats_negative[cfg.steering_type]

    # 5. get feature top activation examples


    # 6. generate with steer
    # steer with positive feature
    generate_with_steering_features(cfg, model, sae, dataset_train, top_k_ind[0], max_act_positive, pos_extract[0])
    # steer with negative feature
    generate_with_steering_features(cfg, model, sae, dataset_train, top_k_ind[1], max_act_negative, pos_extract[1])


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
        pos_extract = [' honest', ' lying']

    if args.task_name == 'honesty':
        contrastive_label = ['honest', 'lying']
    elif args.task_name == 'jailbreak':
        contrastive_label = ['HHH', args.jailbreak]

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 sae_release=args.sae_release, sae_id=args.sae_id,
                 pos_extract=pos_extract, pos_type=args.pos_type,
                 task_name=args.task_name, contrastive_label=contrastive_label)
