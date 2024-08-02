import torch
import os
import math
from tqdm import tqdm
from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import einops
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import pickle
from pipeline.utils.hook_utils import get_and_cache_direction_ablation_input_pre_hook
from pipeline.utils.hook_utils import get_and_cache_diff_addition_input_pre_hook
from pipeline.utils.hook_utils import get_and_cache_direction_ablation_output_hook
from pipeline.utils.hook_utils import get_generation_cache_activation_trajectory_input_pre_hook
from pipeline.utils.hook_utils import get_activations_pre_hook, get_activations_hook
from pipeline.utils.hook_utils import get_generation_cache_activation_input_pre_hook
import plotly.io as pio
from pipeline.submodules.evaluate_truthful import get_performance_stats
from pipeline.analysis.stage_statistics import get_state_quantification
from torch.nn.functional import softmax
from pipeline.submodules.evaluate_truthful_correct_wrong import get_performance_stats
import pickle
import json
from pipeline.analysis.stage_statistics import get_state_quantification


def get_ablation_activations_pre_hook(layer, cache: Float[Tensor, "batch layer d_model"], n_samples, positions: List[int],batch_id,batch_size):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[batch_id:batch_id+batch_size, layer] =  torch.squeeze(activation[:, positions, :],1)
    return hook_fn


def get_addition_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module],
                             direction,
                             batch_size=32, positions=[-1],target_layer=None):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store activations in high-precision to avoid numerical issues
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)

    # if not specified, ablate all layers by default
    if target_layer==None:
        target_layer = np.arange(n_layers)

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        fwd_pre_hooks = [(block_modules[layer],
                          get_and_cache_diff_addition_input_pre_hook(
                                                   direction=direction,
                                                   cache=activations,
                                                   layer=layer,
                                                   positions=positions,
                                                   batch_id=i,
                                                   batch_size=batch_size,
                                                   target_layer=target_layer),
                                                ) for layer in range(n_layers)]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return activations


def generate_get_contrastive_activations_and_plot_pca(cfg, model_base, tokenize_fn,
                                                      dataset_1, dataset_2,
                                                      labels_1=None, labels_2=None,
                                                      save_activations=False, save_plot=False,
                                                      contrastive_name=['correct', 'wrong']):
    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    model_name = cfg.model_alias
    data_category = cfg.data_category
    max_new_tokens = cfg.max_new_tokens
    tokenize_fn_lie = model_base.tokenize_statements_fn
    true_token_id = model_base.true_token_id
    false_token_id = model_base.false_token_id
    # tokenize_fn_syco = model_base.tokenize_statements_fn_syco

    # 1. activation extraction with generation
    activations_1, completions_1, first_gen_toks_1, first_gen_str_1 = generate_and_get_activations(cfg,
                                                                                                   model_base,
                                                                                                   dataset_1,
                                                                                                   tokenize_fn,
                                                                                                   positions=[-1],
                                                                                                   max_new_tokens=max_new_tokens,
                                                                                                   system_type="honest",
                                                                                                   labels=labels_1)

    activations_2, completions_2, first_gen_toks_2, first_gen_str_2 = generate_and_get_activations(
        cfg, model_base, dataset_2,
        tokenize_fn,
        positions=[-1],
        max_new_tokens=max_new_tokens,
        system_type="honest",
        labels=labels_2)

    # 2.1 save activations
    if save_activations:
        activations = {
            "activations_1": activations_1,
            "activations_2": activations_2,
        }
        with open(artifact_dir + os.sep + model_name + '_' + f'{data_category}'
                  + '_activation_pca_' + contrastive_name[0] + '_' + contrastive_name[1] + '.pkl', "wb") as f:
            pickle.dump(activations, f)

    # 2.2 save completions
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    with open(f'{cfg.artifact_path()}' + os.sep + 'completions' + os.sep +
              f'{data_category}_completions_{contrastive_name[0]}.json',
              "w") as f:
        json.dump(completions_1, f, indent=4)
    with open(f'{cfg.artifact_path()}' + os.sep + 'completions' + os.sep +
              f'{data_category}_completions_{contrastive_name[1]}.json',
              "w") as f:
        json.dump(completions_2, f, indent=4)

    # 3. plot pca
    n_layers = model_base.model.config.num_hidden_layers
    fig = plot_contrastive_activation_pca(activations_1, activations_2,
                                          n_layers,
                                          contrastive_name=contrastive_name,
                                          labels_1=labels_1, labels_2=labels_2)
    if save_plot:
          fig.write_html(artifact_dir + os.sep + model_name + '_' + 'activation_pca_' +
                         contrastive_name[0] + '_' + contrastive_name[1] + '.html')

          pio.write_image(fig, artifact_dir + os.sep + model_name + '_' + 'activation_pca_' +
                          contrastive_name[0] + '_' + contrastive_name[1] + '.png',
                          scale=6)

    # 4. get performance
    save_path = artifact_dir + os.sep + "performance"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_performance, fig = get_performance_stats(cfg,
                                                   first_gen_toks_1, first_gen_str_1,
                                                   first_gen_toks_2, first_gen_str_2,
                                                   labels_1, labels_2,
                                                   true_token_id, false_token_id
                                                   )
    fig.write_html(save_path + os.sep + f'{data_category}_performance_' +
                   contrastive_name[0] + '_' + contrastive_name[1] + '.html')
    pio.write_image(fig, save_path + os.sep + f'{data_category}_performance_' +
                    contrastive_name[0] + '_' + contrastive_name[1] + '.png',
                    scale=6)

    # # 5. Get stage statistics
    # save_path = artifact_dir + os.sep + "stage_stats"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # stage_stats = get_state_quantification(cfg, activations_1, activations_lying,
    #                                        labels,
    #                                        save_plot=True)
    # with open(save_path + os.sep + model_name + '_' + f'{data_category}' +
    #           '_stage_stats.pkl', "wb") as f:
    #     pickle.dump(stage_stats, f)
    # return activations_1, activations_lying


def get_addition_activations_generation(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module],
                             direction,
                             batch_size=32, positions=[-1], target_layer=None,
                             max_new_tokens=64):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)

    # if not specified, ablate all layers by default
    if target_layer==None:
        target_layer=np.arange(n_layers)

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    generation_config.pad_token_id = tokenizer.pad_token_id

    completions = []
    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        len_inputs= inputs.input_ids.shape[1]

        fwd_pre_hooks = [(block_modules[layer],
                          get_and_cache_diff_addition_input_pre_hook(
                                                   direction=direction,
                                                   cache=activations,
                                                   layer=layer,
                                                   positions=positions,
                                                   batch_id=i,
                                                   batch_size=batch_size,
                                                   target_layer=target_layer,
                                                   len_prompt=len_inputs),
                                                ) for layer in range(n_layers)]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            generation_toks = model.generate(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                generation_config=generation_config,
            )

            generation_toks = generation_toks[:, inputs.input_ids.shape[-1]:]

            for generation_idx, generation in enumerate(generation_toks):
                completions.append({
                    'prompt': instructions[i + generation_idx],
                    'response': tokenizer.decode(generation, skip_special_tokens=True).strip()
                })

    return activations, completions


def generate_with_cache_trajectory(inputs, model, block_modules,
                                   batch_id,
                                   batch_size,
                                   cache_type="prompt",
                                   positions=-1,
                                   max_new_tokens=64):

    len_prompt = inputs.input_ids.shape[1]
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    all_toks = torch.zeros((inputs.input_ids.shape[0], inputs.input_ids.shape[1] + max_new_tokens),
                           dtype=torch.long,
                           device=model.device)
    all_toks[:, :inputs.input_ids.shape[1]] = inputs.input_ids
    attention_mask = torch.ones((inputs.input_ids.shape[0], inputs.input_ids.shape[1] + max_new_tokens),
                                dtype=torch.long,
                                device=model.device)
    attention_mask[:, :inputs.input_ids.shape[1]] = inputs.attention_mask



    if cache_type == "prompt":
        activations = torch.zeros((batch_size, n_layers, d_model), dtype=torch.float64, device=model.device)
    elif cache_type == "trajectory":
        activations = torch.zeros((batch_size, max_new_tokens, n_layers, d_model), dtype=torch.float64, device=model.device)
    for ii in range(max_new_tokens):
        if cache_type == "prompt":
            cache_position = positions
        elif cache_type == "trajectory":
            cache_position = ii
        fwd_pre_hooks = [(block_modules[layer],
                          get_generation_cache_activation_trajectory_input_pre_hook(activations,
                                                                                    layer,
                                                                                    positions=cache_position,
                                                                                    batch_id=batch_id,
                                                                                    batch_size=batch_size,
                                                                                    cache_type=cache_type,
                                                                                    len_prompt=len_prompt)
                          ) for layer in range(n_layers)]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                logits = model(input_ids=all_toks[:, :-max_new_tokens + ii],
                               attention_mask=attention_mask[:, :-max_new_tokens + ii],)
                next_tokens = logits[0][:, -1, :].argmax(dim=-1)  # greedy sampling (temperature=0)
                all_toks[:, -max_new_tokens + ii] = next_tokens

    generation_toks = all_toks[:, inputs.input_ids.shape[-1]:]
    return generation_toks, activations


def generate_and_get_activation_trajectory(cfg, model_base, dataset,
                               tokenize_fn,
                               positions=[-1],
                               max_new_tokens=64,
                               system_type=None,
                               labels=None,
                               cache_type="trajectory"):

    torch.cuda.empty_cache()

    model_name = cfg.model_alias
    batch_size = 1
    dataset_id = cfg.dataset_id
    model = model_base.model
    block_modules = model_base.model_block_modules
    tokenizer = model_base.tokenizer
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_samples = len([dataset_id])

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    generation_config.pad_token_id = tokenizer.pad_token_id

    completions = []
    for id in tqdm([dataset_id]):
        inputs = tokenize_fn(prompts=dataset[id:id+batch_size], system_type=system_type)

        generation_toks, activations = generate_with_cache_trajectory(inputs, model, block_modules,
                                                                      id,
                                                                      batch_size,
                                                                      cache_type=cache_type,
                                                                      positions=-1,
                                                                      max_new_tokens=max_new_tokens)
        for generation_idx, generation in enumerate(generation_toks):
            if labels is not None:
                completions.append({
                    'prompt': dataset[id + generation_idx],
                    'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                    'label': labels[id + generation_idx],
                    'ID': id+generation_idx
                })
            else:
                completions.append({
                    'prompt': dataset[id + generation_idx],
                    'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                    'ID': id + generation_idx
                })
    return activations, completions


def generate_and_get_activations(cfg, model_base, dataset,
                                tokenize_fn,
                                positions=[-1],
                                max_new_tokens=64,
                                system_type=None,
                                labels=None):

    torch.cuda.empty_cache()

    model_name = cfg.model_alias
    batch_size = cfg.batch_size
    model = model_base.model
    block_modules = model_base.model_block_modules
    tokenizer = model_base.tokenizer
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_samples = len(dataset)

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    generation_config.pad_token_id = tokenizer.pad_token_id

    completions = []
    # we store the mean activations in high-precision to avoid numerical issues
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)
    first_gen_toks_all = torch.zeros((n_samples), dtype=torch.long)
    first_gen_str_all = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        inputs = tokenize_fn(prompts=dataset[i:i+batch_size], system_type=system_type)
        len_prompt = inputs.input_ids.shape[1]
        fwd_pre_hooks = [(block_modules[layer],
                          get_generation_cache_activation_input_pre_hook(activations,
                                                                         layer,
                                                                         positions=-1,
                                                                         batch_id=i,
                                                                         batch_size=batch_size,
                                                                         len_prompt=len_prompt)
                          ) for layer in range(n_layers)]
        #fwd_hook = [module,hook_function]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            generation_toks = model.generate(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                generation_config=generation_config,
            )
            first_gen_tok = generation_toks[:, inputs.input_ids.shape[-1]]
            top_token_str = tokenizer.batch_decode(first_gen_tok)
            first_gen_toks_all[i:i + batch_size] = first_gen_tok
            first_gen_str_all.append(top_token_str)
            generation_toks = generation_toks[:, inputs.input_ids.shape[-1]:]

            for generation_idx, generation in enumerate(generation_toks):
                if labels is not None:
                    completions.append({
                        'prompt': dataset[i + generation_idx],
                        'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                        'label': labels[i + generation_idx],
                        'ID': i+generation_idx
                    })
                else:
                    completions.append({
                        'prompt': dataset[i + generation_idx],
                        'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                        'ID': i + generation_idx
                    })
    first_gen_str_all = [x for xs in first_gen_str_all for x in xs]

    return activations, completions, first_gen_toks_all, first_gen_str_all


def get_activations(cfg, model_base, dataset,
                    tokenize_fn,
                    positions=[-1],
                    system_type=None,
                    intervention=None):

    torch.cuda.empty_cache()
    if intervention is None:
        intervention = cfg.intervention
    model_name = cfg.model_alias
    batch_size = cfg.batch_size
    model = model_base.model
    block_modules = model_base.model_block_modules
    tokenizer = model_base.tokenizer

    n_layers = model.config.num_hidden_layers
    n_samples = len(dataset)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    # activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)
    first_gen_toks_all = torch.zeros((n_samples), dtype=torch.long)
    first_gen_str_all = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        inputs = tokenize_fn(prompts=dataset[i:i+batch_size], system_type=system_type)

        fwd_pre_hooks = []
        if "mlp" in intervention:
            fwd_hooks = [(block_modules[layer].mlp,
                                  get_activations_hook(layer=layer,
                                                       cache=activations,
                                                       positions=positions,
                                                       batch_id=i,
                                                       batch_size=batch_size)) for layer in range(n_layers)]
        elif "attn" in intervention:
            if "Qwen" in model_name:
                fwd_hooks = [(block_modules[layer].attn,
                                  get_activations_hook(layer=layer,
                                                       cache=activations,
                                                       positions=positions,
                                                       batch_id=i,
                                                       batch_size=batch_size)) for layer in range(n_layers)]
            else:
                fwd_hooks = [(block_modules[layer].self_attn,
                                  get_activations_hook(layer=layer,
                                                       cache=activations,
                                                       positions=positions,
                                                       batch_id=i,
                                                       batch_size=batch_size)) for layer in range(n_layers)]
        else:
            fwd_hooks = [(block_modules[layer],
                                  get_activations_hook(layer=layer,
                                                       cache=activations,
                                                       positions=positions,
                                                       batch_id=i,
                                                       batch_size=batch_size)) for layer in range(n_layers)]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            outputs =  model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            ).logits[:, -1, :]
            probs = softmax(outputs, dim=-1)

            # Sort the probabilities in descending order and get the top tokens
            top_probs, top_indices = torch.topk(probs, k=5)  # Get top 5 for demonstration

            # Get the top token and its probability
            top_token_id = top_indices[:, 0]
            top_token_str = tokenizer.batch_decode(top_token_id)

            first_gen_toks_all[i:i+batch_size] = top_token_id
            first_gen_str_all.append(top_token_str)
    first_gen_str_all = [x for xs in first_gen_str_all for x in xs]
    return activations, first_gen_toks_all, first_gen_str_all


def get_contrastive_activations_and_plot_pca(cfg,
                                             model_base,
                                             dataset,
                                             labels=None,
                                             intervention=None,
                                             save_activations=False,
                                             save_plot=True
                                             ):

    if intervention is None:
        intervention = cfg.intervention
    model_name = cfg.model_alias
    data_category = cfg.data_category
    artifact_dir = cfg.artifact_path()
    true_token_id = model_base.true_token_id
    false_token_id = model_base.false_token_id

    if not os.path.exists(os.path.join(artifact_dir, intervention)):
        os.makedirs(os.path.join(artifact_dir, intervention))

    tokenize_fn = model_base.tokenize_statements_fn
    activations_lying, first_gen_toks_lying, first_gen_str_lying = get_activations(cfg, model_base, dataset,
                                        tokenize_fn,
                                        positions=[-1],
                                        system_type="lying",
                                        intervention=intervention)

    activations_1, first_gen_toks_honest, first_gen_str_honest = get_activations(cfg, model_base, dataset,
                                         tokenize_fn,
                                         positions=[-1],
                                         system_type="honest",
                                         intervention=intervention)

    # save activations
    if save_activations:
        activations = {
            "activations_1": activations_1,
            "activations_lying": activations_lying,
        }
        with open(artifact_dir + os.sep + intervention + os.sep + model_name + '_' + f'{data_category}'
                       + '_activation_pca.pkl', "wb") as f:
            pickle.dump(activations, f)

    # 2. plot and save pca plots
    if save_plot:
        n_layers = model_base.model.config.num_hidden_layers
        fig = plot_contrastive_activation_pca(activations_1, activations_lying,
                                              n_layers, contrastive_name=["honest", "lying"],
                                              labels=labels)
        fig.write_html(artifact_dir + os.sep + intervention + os.sep + f'{data_category}'
                        + '_activation_pca.html')
        pio.write_image(fig, artifact_dir + os.sep + intervention + os.sep + f'{data_category}'
                        + '_activation_pca.pdf')
        pio.write_image(fig, artifact_dir + os.sep + intervention + os.sep + f'{data_category}'
                        + '_activation_pca.png',
                        scale=6)

    # 3. get performance
    save_path = artifact_dir + os.sep + "performance"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_performance, fig = get_performance_stats(cfg, first_gen_toks_honest, first_gen_str_honest, first_gen_toks_lying, first_gen_str_lying,
                                                   labels,
                                                   true_token_id, false_token_id
                                                   )
    fig.write_html(save_path + os.sep + f'{data_category}_{intervention}_'
                   + 'model_performance.html')
    pio.write_image(fig, save_path + os.sep + f'{data_category}_{intervention}_'
                   + 'model_performance.png',
                    scale=6)
    # 4. quantify different lying stages
    stage_stats = get_state_quantification(cfg, activations_1, activations_lying, labels,
                                           save_plot=False)

    return activations_1, activations_lying, model_performance, stage_stats


def plot_contrastive_activation_pca_with_trajectory(activations_1, activations_2,
                                                    activation_trajectory_honest, activation_trajectory_lie,
                                                    n_layers,
                                                    str_honest, str_lie,
                                                    contrastive_name=["honest", "lying", "trajectory_honest", "trajectory_lying"],
                                                    labels=None):
    n_contrastive_data = activations_2.shape[0]
    n_trajectory_honest = len(str_honest)
    n_trajectory_lie = len(str_lie)
    # only take the part of the answer that is actually generated (otherwise the 0 will mess up pca and produce nan values)
    activation_trajectory_lie = activation_trajectory_lie[:, :n_trajectory_lie, :, :]
    activation_trajectory_honest = activation_trajectory_honest[:, :n_trajectory_honest, :, :]
    # reshape the activation trajectory to to the same format as the activations
    activation_trajectory_lie = einops.rearrange(activation_trajectory_lie, '1 seq n_layers d_model -> seq n_layers d_model')
    labels_trajectory_lie = [2+0*i for i in range(activation_trajectory_lie.shape[0])]
    activation_trajectory_honest = einops.rearrange(activation_trajectory_honest, '1 seq n_layers d_model -> seq n_layers d_model')
    labels_trajectory_honest = [3+0*i for i in range(activation_trajectory_honest.shape[0])]
    # concatenate all
    # activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_1,
    #                                                                           activations_2,
    #                                                                           activation_trajectory_honest,
    #                                                                           activation_trajectory_lie), dim=0)
    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_1,
                                                                              activations_2), dim=0)
    activations_all_trajectory = torch.cat((activation_trajectory_honest,
                                            activation_trajectory_lie), dim=0)
    if labels is not None:
        labels_all = labels + labels + labels_trajectory_honest + labels_trajectory_lie
        # true or false label
        labels_tf = []
        for ll in labels:
            if ll == 0:
                labels_tf.append('false')
            elif ll == 1:
                labels_tf.append('true')

    label_text = []
    for ii in range(n_contrastive_data):
        label_text = np.append(label_text, f'{contrastive_name[0]}_{labels_tf[ii]}_{ii}')
    for ii in range(n_contrastive_data):
        label_text = np.append(label_text, f'{contrastive_name[1]}_{labels_tf[ii]}_{ii}')
    for ii in range(n_trajectory_honest):
        label_text = np.append(label_text, f'{contrastive_name[2]}_{labels_tf[ii]}_{ii}')
    for ii in range(n_trajectory_lie):
        label_text = np.append(label_text, f'{contrastive_name[3]}_{labels_tf[ii]}_{ii}')
    cols = 4
    rows = math.ceil(n_layers/cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"layer {n}" for n in range(n_layers)])

    pca = PCA(n_components=3)
    for row in range(rows):
        for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
            # print(f'layer{layer}')
            if layer < n_layers:
                activations_pca = pca.fit_transform(activations_all[:, layer, :].cpu())
                activations_pca_trajectory = pca.transform(activations_all_trajectory[:, layer, :].cpu())
                activations_pca_all = np.concatenate((activations_pca, activations_pca_trajectory), axis=0)

                # activations = np.concatenate((activations_all[:, layer, :].cpu(), activations_all_trajectory[:, layer, :].cpu()), axis=0)
                # activations_pca_all = pca.fit_transform(activations)

                df = {}
                df['label'] = labels_all
                df['pca0'] = activations_pca_all[:, 0]
                df['pca1'] = activations_pca_all[:, 1]
                df['label_text'] = label_text

                # plot honest data
                fig.add_trace(
                    go.Scatter(x=df['pca0'][:n_contrastive_data],
                               y=df['pca1'][:n_contrastive_data],
                               mode="markers",
                               showlegend=False,
                               marker=dict(
                                   symbol="star",
                                   size=8,
                                   line=dict(width=1, color="DarkSlateGrey"),
                                   color=df['label'][:n_contrastive_data]
                               ),
                           text=df['label_text'][:n_contrastive_data]),
                           row=row+1, col=ll+1,
                            )
                # plot lying data
                fig.add_trace(
                    go.Scatter(x=df['pca0'][n_contrastive_data:n_contrastive_data*2],
                               y=df['pca1'][n_contrastive_data:n_contrastive_data*2],
                               mode="markers",
                               showlegend=False,
                               marker=dict(
                                   symbol="circle",
                                   size=5,
                                   line=dict(width=1, color="DarkSlateGrey"),
                                   color=df['label'][n_contrastive_data:n_contrastive_data*2],
                               ),
                           text=df['label_text'][n_contrastive_data:n_contrastive_data*2]),
                           row=row+1, col=ll+1,
                           )

                #plot honest trajectory
                fig.add_trace(
                    go.Scatter(x=df['pca0'][n_contrastive_data*2:n_contrastive_data*2+n_trajectory_honest],
                               y=df['pca1'][n_contrastive_data*2:n_contrastive_data*2+n_trajectory_honest],
                               mode="markers",
                               showlegend=False,
                               marker=dict(
                                   symbol="square",
                                   size=5,
                                   line=dict(width=1, color="DarkSlateGrey"),
                                   color="green",
                               ),
                           text=df['label_text'][n_contrastive_data*2:n_contrastive_data*2+n_trajectory_honest]),
                           row=row+1, col=ll+1,
                           )
                # plot lying trajectory
                fig.add_trace(
                    go.Scatter(x=df['pca0'][-n_trajectory_lie:],
                               y=df['pca1'][-n_trajectory_lie:],
                               mode="markers",
                               showlegend=False,
                               marker=dict(
                                   symbol="triangle-up",
                                   size=5,
                                   line=dict(width=1, color="DarkSlateGrey"),
                                   color="red",
                               ),
                           text=df['label_text'][-n_trajectory_lie:]),
                           row=row+1, col=ll+1,
                           )
    # legend
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       symbol="star",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                   ),
                   name=f'honest_false',
                   marker_color='blue',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       symbol="star",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                   ),
                   name=f'honest_true',
                   marker_color='yellow',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       symbol="circle",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                   ),
                   name=f'lying_false',
                   marker_color='blue',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       symbol="circle",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                   ),
                   name=f'lying_true',
                   marker_color='yellow',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       symbol="square",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                   ),
                   name=f'trajectory_hoenst',
                   marker_color='red',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode="markers",
                   marker=dict(
                       symbol="triangle-up",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                   ),
                   name=f'trajectory_lying',
                   marker_color='red',
                    ),
        row=row + 1, col=ll + 1,
    )
    fig.update_layout(height=1600, width=1000)
    fig.show()
    fig.write_html('honest_lying_pca.html')

    return fig


def plot_contrastive_activation_pca(activations_1, activations_2,
                                    n_layers, contrastive_name,
                                    labels_1=None, labels_2=None,
                                    probability_1=None, probability_2=None,
                                    color_by='label'):

    n_contrastive_data = activations_2.shape[0]

    if color_by == 'label':
        if labels_1 is not None and labels_2 is None:
            labels_all = labels_1 + labels_1
            labels_tf = []
            for ll in labels_1:
                if ll == 0:
                    labels_tf.append('false')
                elif ll == 1:
                    labels_tf.append('true')
            for ll in labels_2:
                if ll == 0:
                    labels_tf.append('false')
                elif ll == 1:
                    labels_tf.append('true')
        elif labels_1 is not None and labels_2 is not None:
            labels_all = labels_1 + labels_2
            # true or false label
            labels_tf = []
            for ll in labels_1:
                if ll == 0:
                    labels_tf.append('false')
                elif ll == 1:
                    labels_tf.append('true')
            for ll in labels_1:
                if ll == 0:
                    labels_tf.append('false')
                elif ll == 1:
                    labels_tf.append('true')
    elif color_by == "probability":
        labels_all = probability_1 + probability_2
        labels_tf = []
        for ll in labels_1:
            if ll == 0:
                labels_tf.append('false')
            elif ll == 1:
                labels_tf.append('true')
        for ll in labels_2:
            if ll == 0:
                labels_tf.append('false')
            elif ll == 1:
                labels_tf.append('true')

    label_text = []
    for ii in range(n_contrastive_data):
        label_text = np.append(label_text, f'{contrastive_name[0]}_{labels_tf[ii]}_{ii}_{probability_1[ii]}')
    for ii in range(n_contrastive_data):
        label_text = np.append(label_text, f'{contrastive_name[1]}_{labels_tf[ii+n_contrastive_data]}_{ii}_{probability_2[ii]}')

    cols = 4
    rows = math.ceil(n_layers/cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"layer {n}" for n in range(n_layers)])

    pca = PCA(n_components=3)
    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_1,
                                                                              activations_2), dim=0)
    for row in range(rows):
        for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
            # print(f'layer{layer}')
            if layer < n_layers:
                # activations_pca_2 = pca.fit_transform(activations_2[:, layer, :].cpu())
                # activations_pca_1 = pca.transform(activations_1[:, layer, :].cpu())
                # activations_pca = np.concatenate((activations_pca_1,
                #                                   activations_pca_2), axis=0)
                activations_pca = pca.fit_transform(activations_all[:, layer, :].cpu())
                df = {}
                df['label'] = labels_all
                df['pca0'] = activations_pca[:, 0]
                df['pca1'] = activations_pca[:, 1]
                df['pca2'] = activations_pca[:, 2]
                df['label_text'] = label_text

                fig.add_trace(
                    go.Scatter(x=df['pca0'][:n_contrastive_data],
                               y=df['pca1'][:n_contrastive_data],
                               # z=df['pca2'][:n_contrastive_data],
                               mode="markers",
                               name="honest",
                               showlegend=False,
                               marker=dict(
                                   symbol="star",
                                   size=8,
                                   line=dict(width=1, color="DarkSlateGrey"),
                                   color=df['label'][:n_contrastive_data],
                                   colorscale="Viridis",
                               ),
                               text=df['label_text'][:n_contrastive_data]
                               ),
                               row=row+1, col=ll+1,
                               )

                # fig.add_trace(
                #     go.Scatter(x=df['pca0'][n_contrastive_data:],
                #                  y=df['pca1'][n_contrastive_data:],
                #                  # z=df['pca2'][n_contrastive_data:],
                #                  mode="markers",
                #                  name="lying",
                #                  showlegend=False,
                #                  marker=dict(
                #                    symbol="circle",
                #                    size=5,
                #                    line=dict(width=1, color="DarkSlateGrey"),
                #                    color=df['label'][n_contrastive_data:],
                #                  ),
                #                  text=df['label_text'][n_contrastive_data:]),
                #     row=row+1, col=ll+1,
                #                  )
    # legend
    fig.add_trace(
        go.Scatter(x=[None],
                     y=[None],
                     # z=[None],
                     mode='markers',
                     marker=dict(
                       symbol="star",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                       color=df['label'][n_contrastive_data:],
                     ),
                     name=f'honest_false',
                     marker_color='blue',
                     ),
        row=row + 1, col=ll + 1,
    )

    fig.add_trace(
        go.Scatter(x=[None],
                     y=[None],
                     # z=[None],
                     mode='markers',
                     marker=dict(
                       symbol="star",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                       color=df['label'][n_contrastive_data:],
                     ),
                     name=f'honest_true',
                     marker_color='yellow',
                     ),
        row=row + 1, col=ll + 1,
    )

    fig.add_trace(
        go.Scatter(x=[None],
                     y=[None],
                     # z=[None],
                     mode='markers',
                     marker=dict(
                       symbol="circle",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                       color=df['label'][n_contrastive_data:],
                     ),
                     name=f'lying_false',
                     marker_color='blue',
                     ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   # z=[None],
                   mode='markers',
                   marker=dict(
                       symbol="circle",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                       color=df['label'][n_contrastive_data:],
                   ),
                   name=f'lying_true',
                   marker_color='yellow',
                   ),
        row=row + 1, col=ll + 1,
    )
    fig.update_layout(height=1600, width=1000)
    fig.show()
    fig.write_html('honest_lying_pca.html')

    return fig


def plot_contrastive_activation_intervention_pca(activations_1,
                                                 activations_2,
                                                 ablation_activations_1,
                                                 ablation_activations_2,
                                                 n_layers,
                                                 contrastive_name,
                                                 labels=None,
                                                 ):

    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_1,
                                                                              activations_2,
                                                                              ablation_activations_1,
                                                                              ablation_activations_2),
                                                                              dim=0)

    pca = PCA(n_components=3)

    label_text = []
    label_plot = []

    for ii in range(activations_1.shape[0]):
        label_text = np.append(label_text, f'{contrastive_name[0]}:{ii}')
        label_plot.append(0)
    for ii in range(activations_2.shape[0]):
        label_text = np.append(label_text, f'{contrastive_name[1]}:{ii}')
        label_plot.append(1)
    for ii in range(activations_1.shape[0]):
        label_text = np.append(label_text, f'{contrastive_name[2]}:{ii}')
        label_plot.append(2)
    for ii in range(activations_1.shape[0]):
        label_text = np.append(label_text, f'{contrastive_name[3]}:{ii}')
        label_plot.append(3)
    cols = 4
    rows = int(n_layers/cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"layer {n}" for n in range(n_layers)])
    for row in range(rows):
        # print(f'row:{row}')
        for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
            # print(f'layer{layer}')
            if layer <= n_layers:
                activations_pca = pca.fit_transform(activations_all[:, layer, :].cpu())
                df = {}
                df['label'] = label_plot
                df['pca0'] = activations_pca[:, 0]
                df['pca1'] = activations_pca[:, 1]
                df['label_text'] = label_text

                fig.add_trace(
                    go.Scatter(x=df['pca0'],
                               y=df['pca1'],
                               mode='markers',
                               marker_color=df['label'],
                               text=df['label_text']),
                    row=row + 1, col=ll + 1,
                )
    # legend
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   name=f'{contrastive_name[0]}',
                   marker_color='blue',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   name=f'{contrastive_name[1]}',
                   marker_color='purple',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   name=f'{contrastive_name[2]}',
                   marker_color='orange',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   name=f'{contrastive_name[3]}',
                   marker_color='yellow',
                 ),
        row=row + 1, col=ll + 1,
    )

    fig.update_layout(
        showlegend=True
    )
    fig.update_layout(height=1600, width=1000)
    fig.show()
    return fig

#
# def extraction_intervention_and_plot_pca(cfg,model_base: ModelBase, harmful_instructions, harmless_instructions):
#     artifact_dir = cfg.artifact_path()
#     if not os.path.exists(artifact_dir):
#         os.makedirs(artifact_dir)
#
#     # 1. extract activations
#     activations_1,activations_2 = generate_activations_and_plot_pca(cfg,model_base, harmful_instructions, harmless_instructions)
#
#     # 2. get steering vector = get mean difference of the source layer
#     mean_activation_harmful = activations_1.mean(dim=0)
#     mean_activation_harmless = activations_2.mean(dim=0)
#     mean_diff = mean_activation_harmful-mean_activation_harmless
#
#
#     # 3. refusal_intervention_and_plot_pca
#     ablation_activations_1,ablation_activations_2 = refusal_intervention_and_plot_pca(cfg, model_base,
#                                                                                              harmful_instructions,
#                                                                                              harmless_instructions,
#                                                                                              mean_diff)
#
#     # 4. pca with and without intervention, plot and save pca
#     intervention = cfg.intervention
#     source_layer = cfg.source_layer
#     target_layer = cfg.target_layer
#     model_name = cfg.model_alias
#     n_layers = model_base.model.config.num_hidden_layers
#     fig = plot_contrastive_activation_intervention_pca(activations_1, activations_2,
#                                                        ablation_activations_1, ablation_activations_2,
#                                                        n_layers)
#     fig.write_html(artifact_dir + os.sep + model_name + '_' + 'refusal_generation_activation_'
#                    +intervention+'_pca_layer_'
#                    +str(source_layer)+'_'+ str(target_layer)+'.html')