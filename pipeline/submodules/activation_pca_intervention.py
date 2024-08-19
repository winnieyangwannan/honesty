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
import plotly.express as px
import plotly.io as pio
from pipeline.utils.hook_utils import get_and_cache_direction_ablation_input_pre_hook
from pipeline.utils.hook_utils import get_and_cache_diff_addition_input_pre_hook, get_and_cache_diff_addition_output_hook
from pipeline.utils.hook_utils import get_and_cache_direction_ablation_output_hook
from pipeline.utils.hook_utils import get_and_cache_skip_connection_input_pre_hook, get_and_cache_skip_connection_hook
from pipeline.utils.hook_utils import get_and_cache_direction_addition_output_hook, get_and_cache_direction_addition_input_hook
from pipeline.submodules.evaluate_truthful import get_performance_stats
import pickle
import json
from pipeline.analysis.stage_statistics import get_state_quantification
from pipeline.submodules.evaluate_jailbreak import evaluate_completions_and_save_results_for_dataset
from pipeline.utils.hook_utils import get_and_cache_direction_projection_output_hook


def plot_contrastive_activation_intervention_pca(activations_positive,
                                                 activations_negative,
                                                 intervention_activations_positive,
                                                 intervention_activations_negative,
                                                 n_layers,
                                                 contrastive_label,
                                                 labels_ori=None,
                                                 labels_int=None,
                                                 plot_original=True,
                                                 plot_intervention=True,
                                                 prompt_label=['true', 'false']
                                                 ):
    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_positive,
                                                                              activations_negative),
                                                                              dim=0)
    activations_all_intervention = torch.cat((
                                              intervention_activations_positive,
                                              intervention_activations_negative),
                                              dim=0)
    pca = PCA(n_components=3)

    label_text = []
    label_plot = []
    n_data_ori = activations_positive.shape[0]
    n_data_int = intervention_activations_positive.shape[0]
    # true or false label
    labels_tf_ori = []
    for ll in labels_ori:
        if ll == 0:
            labels_tf_ori.append(prompt_label[1])
        elif ll == 1:
            labels_tf_ori.append(prompt_label[0])
    labels_tf_int = []
    for ll in labels_int:
        if ll == 0:
            labels_tf_int.append(prompt_label[1])
        elif ll == 1:
            labels_tf_int.append(prompt_label[0])

    for ii in range(n_data_ori):
        label_text = np.append(label_text, f'{contrastive_label[0]}_{labels_tf_ori[ii]}:{ii}')
        # label_plot.append(0)
    for ii in range(n_data_ori):
        label_text = np.append(label_text, f'{contrastive_label[1]}_{labels_tf_ori[ii]}:{ii}')
        # label_plot.append(1)
    for ii in range(n_data_int):
        label_text = np.append(label_text, f'{contrastive_label[2]}_{labels_tf_int[ii]}:{ii}')
        # label_plot.append(2)
    for ii in range(n_data_int):
        label_text = np.append(label_text, f'{contrastive_label[3]}_{labels_tf_int[ii]}:{ii}')
        # label_plot.append(3)
    label_plot = labels_ori + labels_ori + labels_int + labels_int
    cols = 4
    rows = int(n_layers/cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"layer {n}" for n in range(n_layers)])
    for row in range(rows):
        # print(f'row:{row}')
        for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
            # print(f'layer{layer}')
            if layer <= n_layers:
                # project to the original honest and lying space
                activations_pca = pca.fit_transform(activations_all[:, layer, :].cpu())
                activations_pca_intervention = pca.transform(activations_all_intervention[:, layer, :].cpu())
                activation_pca_all = np.concatenate((activations_pca, activations_pca_intervention), axis=0)

                # project to the intervened honest and lying space
                # activations_pca_intervention = pca.fit_transform(activations_all_intervention[:, layer, :].cpu())
                # activations_pca = pca.transform(activations_all[:, layer, :].cpu())
                # activation_pca_all = np.concatenate((activations_pca, activations_pca_intervention), axis=0)

                df = {}
                df['label'] = label_plot
                df['pca0'] = activation_pca_all[:, 0]
                df['pca1'] = activation_pca_all[:, 1]
                df['label_text'] = label_text

                if plot_original:
                    fig.add_trace(
                        go.Scatter(x=df['pca0'][:n_data_ori],
                                   y=df['pca1'][:n_data_ori],
                                   mode='markers',
                                   showlegend=False,
                                   marker=dict(
                                       symbol="star-open",
                                       size=8,
                                       line=dict(width=2, color="DarkSlateGrey"),
                                       color=df['label'][:n_data_ori],
                                   ),
                                   text=df['label_text'][:n_data_ori]),
                        row=row + 1, col=ll + 1,
                    )

                    fig.add_trace(
                        go.Scatter(x=df['pca0'][n_data_ori:n_data_ori*2],
                                   y=df['pca1'][n_data_ori:n_data_ori*2],
                                   mode='markers',
                                   showlegend=False,
                                   marker=dict(
                                       symbol="circle-open",
                                       size=8,
                                       line=dict(width=2, color="DarkSlateGrey"),
                                       color=df['label'][n_data_ori:n_data_ori*2],
                                   ),
                                   text=df['label_text'][n_data_ori:n_data_ori*2]),
                        row=row + 1, col=ll + 1,
                    )
                if plot_intervention:
                    fig.add_trace(
                        go.Scatter(x=df['pca0'][n_data_ori*2:n_data_ori*2+n_data_int],
                                   y=df['pca1'][n_data_ori*2:n_data_ori*2+n_data_int],
                                   mode='markers',
                                   showlegend=False,
                                   marker=dict(
                                       symbol="star",
                                       size=8,
                                       line=dict(width=1, color="DarkSlateGrey"),
                                       color=df['label'][n_data_ori*2:n_data_ori*2+n_data_int],
                                   ),
                                   text=df['label_text'][n_data_ori*2:n_data_ori*2+n_data_int]),
                        row=row + 1, col=ll + 1,
                    )
                    fig.add_trace(
                        go.Scatter(x=df['pca0'][-n_data_int:],
                                   y=df['pca1'][-n_data_int:],
                                   mode='markers',
                                   showlegend=False,
                                   marker=dict(
                                       symbol="circle",
                                       size=8,
                                       line=dict(width=1, color="DarkSlateGrey"),
                                       color=df['label'][-n_data_int:],
                                   ),
                                   text=df['label_text'][-n_data_int:]),
                        row=row + 1, col=ll + 1,
                    )
    # legend
    if plot_original:
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="star",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           colorscale='PiYG',
                           color=1
                       ),
                       name=f'{contrastive_label[0]}_{prompt_label[0]}',
                       ),
        )

        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="star",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color=1,
                           colorscale='PiYG',
                       ),
                       name=f'{contrastive_label[0]}_{prompt_label[1]}',
                       ),
        )
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="circle",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           colorscale='PiYG',
                           color=1
                       ),
                       name=f'{contrastive_label[1]}_{prompt_label[0]}',
                       ),
        )
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="circle",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           colorscale='PiYG',
                           color=0
                       ),
                       name=f'{contrastive_label[1]}_{prompt_label[1]}',
                       ),
        )
    if plot_intervention:
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="star",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color='yellow'
                       ),
                       name=f'{contrastive_label[0]}_intervention_{prompt_label[0]}',
                       ),
        )
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="star",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color='blue'
                       ),
                       name=f'{contrastive_label[0]}_intervention_{prompt_label[1]}',
                       ),
        )
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="circle",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color='yellow'
                       ),
                       name=f'{contrastive_label[1]}_intervention_{prompt_label[0]}',
                       ),
        )
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="circle",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color='blue'

                       ),
                       name=f'{contrastive_label[1]}_intervention_{prompt_label[1]}',
                       ),
        )

    fig.update_layout(height=1600, width=1000)
    fig.show()
    return fig


def get_intervention_activations_and_generation(cfg, model_base, dataset,
                                                tokenize_fn,
                                                positions=[-1],
                                                mean_diff=None,
                                                target_layer_s=None,
                                                target_layer_e=None,
                                                max_new_tokens=64,
                                                system_type=None,
                                                labels=None,
                                                categories=None):
    """
    output:
        activations: Tensor[batch,layer,1],
        completions: {'prompt': dataset[i + generation_idx],
                        'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                        'label': labels[i + generation_idx],
                        'ID': i + generation_idx}
        first_gen_toks_all: Tensor[batch,1] --> token id for the first token of the generated answer (for accuracy quantification)
        first_gen_str_all: List[str] --> decoded str for the first token of the  generated answer (for accuracy quantification)
    """
    torch.cuda.empty_cache()

    batch_size = cfg.batch_size
    intervention = cfg.intervention
    model = model_base.model
    model_name = cfg.model_alias
    block_modules = model_base.model_block_modules
    tokenizer = model_base.tokenizer
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_samples = len(dataset)

    # if not specified, ablate all layers by default
    if target_layer_s is None:
        target_layer = np.arange(n_layers)
    # apply intervention on one single layer
    elif type(target_layer_s) == int and target_layer_e is None:
        target_layer = [np.arange(n_layers)[target_layer_s]]
    # apply intervention on a range of layers
    elif type(target_layer_s) == int and type(target_layer_e) == int:
        target_layer = np.arange(target_layer_s, target_layer_e)

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    generation_config.pad_token_id = tokenizer.pad_token_id

    completions = []
    first_gen_toks_all = torch.zeros((n_samples), dtype=torch.long)
    first_gen_str_all = []
    # we store the mean activations in high-precision to avoid numerical issues
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)
    for i in tqdm(range(0, len(dataset), batch_size)):
        inputs = tokenize_fn(prompts=dataset[i:i+batch_size], system_type=system_type)
        len_inputs = inputs.input_ids.shape[1]
        if 'positive_projection' in intervention:
            fwd_pre_hooks = []
            fwd_hooks = [(block_modules[layer],
                          get_and_cache_direction_projection_output_hook(
                              mean_diff=mean_diff,
                              cache=activations,
                              layer=layer,
                              positions=positions,
                              batch_id=i,
                              batch_size=batch_size,
                              target_layer=target_layer,
                              len_prompt=len_inputs
                          ),
                          ) for layer in range(n_layers)]

        if 'direction_ablation' in intervention:
            fwd_pre_hooks = []
            if "mlp" in intervention:
                fwd_hooks = [(block_modules[layer].mlp,
                              get_and_cache_direction_ablation_output_hook(
                                  mean_diff=mean_diff,
                                  cache=activations,
                                  layer=layer,
                                  positions=positions,
                                  batch_id=i,
                                  batch_size=batch_size,
                                  target_layer=target_layer,
                                  len_prompt=len_inputs
                              ),
                              ) for layer in range(n_layers)]
            if "attn" in intervention:
                fwd_hooks = [(block_modules[layer].attn,
                              get_and_cache_direction_ablation_output_hook(
                                  mean_diff=mean_diff,
                                  cache=activations,
                                  layer=layer,
                                  positions=positions,
                                  batch_id=i,
                                  batch_size=batch_size,
                                  target_layer=target_layer,
                                  len_prompt=len_inputs
                              ),
                              ) for layer in range(n_layers)]

            else:
                fwd_hooks = [(block_modules[layer],
                              get_and_cache_direction_ablation_output_hook(
                                  mean_diff=mean_diff,
                                  cache=activations,
                                  layer=layer,
                                  positions=positions,
                                  batch_id=i,
                                  batch_size=batch_size,
                                  target_layer=target_layer,
                                  len_prompt=len_inputs
                              ),
                              ) for layer in range(n_layers)]

        if 'direction_addition' in intervention:
            if "mlp" in intervention:
                # fwd_pre_hooks = [(block_modules[layer].mlp,
                #                   get_and_cache_direction_addition_input_hook(
                #                       mean_diff=mean_diff,
                #                       cache=activations,
                #                       layer=layer,
                #                       positions=positions,
                #                       batch_id=i,
                #                       batch_size=batch_size,
                #                       target_layer=target_layer,
                #                       len_prompt=len_inputs),
                #                   ) for layer in range(n_layers)]
                fwd_pre_hooks = []
                fwd_hooks = [(block_modules[layer].mlp,
                              get_and_cache_direction_addition_output_hook(
                                  mean_diff=mean_diff,
                                  cache=activations,
                                  layer=layer,
                                  positions=positions,
                                  batch_id=i,
                                  batch_size=batch_size,
                                  target_layer=target_layer,
                                  len_prompt=len_inputs),
                              ) for layer in range(n_layers)]
            elif "attn" in intervention:
                if "Qwen" in model_name:
                    fwd_pre_hooks = []
                    fwd_hooks = [(block_modules[layer].attn,
                                  get_and_cache_direction_addition_output_hook(
                                      mean_diff=mean_diff,
                                      cache=activations,
                                      layer=layer,
                                      positions=positions,
                                      batch_id=i,
                                      batch_size=batch_size,
                                      target_layer=target_layer,
                                      len_prompt=len_inputs),
                                  ) for layer in range(n_layers)]
                else:
                    fwd_pre_hooks = []
                    fwd_hooks = [(block_modules[layer].self_attn,
                                  get_and_cache_direction_addition_output_hook(
                                      mean_diff=mean_diff,
                                      cache=activations,
                                      layer=layer,
                                      positions=positions,
                                      batch_id=i,
                                      batch_size=batch_size,
                                      target_layer=target_layer,
                                      len_prompt=len_inputs),
                                  ) for layer in range(n_layers)]
            else:
                fwd_pre_hooks = []
                fwd_hooks = [(block_modules[layer],
                              get_and_cache_direction_addition_output_hook(
                                  mean_diff=mean_diff,
                                  cache=activations,
                                  layer=layer,
                                  positions=positions,
                                  batch_id=i,
                                  batch_size=batch_size,
                                  target_layer=target_layer,
                                  len_prompt=len_inputs),
                              ) for layer in range(n_layers)]

        elif "addition" in intervention:
            fwd_pre_hooks = []
            if "mlp" in intervention:
                fwd_hooks = [(block_modules[layer].mlp,
                              get_and_cache_diff_addition_output_hook(
                                                       mean_diff=mean_diff,
                                                       cache=activations,
                                                       layer=layer,
                                                       positions=positions,
                                                       batch_id=i,
                                                       batch_size=batch_size,
                                                       target_layer=target_layer,
                                                       len_prompt=len_inputs),
                                                    ) for layer in range(n_layers)]
            elif "attn" in intervention:
                if "Llama" in model_name or "gemma" in model_name or "Yi" in model_name:
                    fwd_hooks = [(block_modules[layer].self_attn,
                                  get_and_cache_diff_addition_output_hook(
                                                           mean_diff=mean_diff,
                                                           cache=activations,
                                                           layer=layer,
                                                           positions=positions,
                                                           batch_id=i,
                                                           batch_size=batch_size,
                                                           target_layer=target_layer,
                                                           len_prompt=len_inputs),
                                                        ) for layer in range(n_layers)]
                elif "Qwen" in model_name:

                    fwd_hooks = [(block_modules[layer].attn,
                                  get_and_cache_diff_addition_output_hook(
                                                           mean_diff=mean_diff,
                                                           cache=activations,
                                                           layer=layer,
                                                           positions=positions,
                                                           batch_id=i,
                                                           batch_size=batch_size,
                                                           target_layer=target_layer,
                                                           len_prompt=len_inputs),
                                                        ) for layer in range(n_layers)]
            else:
                fwd_hooks = [(block_modules[layer],
                              get_and_cache_diff_addition_output_hook(
                                                       mean_diff=mean_diff,
                                                       cache=activations,
                                                       layer=layer,
                                                       positions=positions,
                                                       batch_id=i,
                                                       batch_size=batch_size,
                                                       target_layer=target_layer,
                                                       len_prompt=len_inputs),
                                                    ) for layer in range(n_layers)]

            # fwd_hooks = []
        elif "skip_connection" in intervention:
            fwd_pre_hooks = []
            if "mlp" in intervention:
                fwd_hooks = [(block_modules[layer].mlp,
                              get_and_cache_skip_connection_hook(
                                                                 mean_diff=mean_diff,
                                                                 cache=activations,
                                                                 layer=layer,
                                                                 positions=positions,
                                                                 batch_id=i,
                                                                 batch_size=batch_size,
                                                                 target_layer=target_layer,
                                                                 len_prompt=len_inputs),
                                                              ) for layer in range(n_layers)]
            elif "attn" in intervention:
                if "Llama" in model_name or "gemma" in model_name or "Yi" in model_name:
                    fwd_hooks = [(block_modules[layer].self_attn,
                                  get_and_cache_skip_connection_hook(
                                                                      mean_diff=mean_diff,
                                                                      cache=activations,
                                                                      layer=layer,
                                                                      positions=positions,
                                                                      batch_id=i,
                                                                      batch_size=batch_size,
                                                                      target_layer=target_layer,
                                                                      len_prompt=len_inputs),
                                                                    ) for layer in range(n_layers)]

                elif "Qwen" in model_name:
                    fwd_hooks = [(block_modules[layer].attn,
                                  get_and_cache_skip_connection_hook(
                                                                      mean_diff=mean_diff,
                                                                      cache=activations,
                                                                      layer=layer,
                                                                      positions=positions,
                                                                      batch_id=i,
                                                                      batch_size=batch_size,
                                                                      target_layer=target_layer,
                                                                      len_prompt=len_inputs),
                                                                    ) for layer in range(n_layers)]
            else:
                fwd_pre_hooks = []
                fwd_hooks = [(block_modules[layer],
                              get_and_cache_skip_connection_hook(
                                                                  mean_diff=mean_diff,
                                                                  cache=activations,
                                                                  layer=layer,
                                                                  positions=positions,
                                                                  batch_id=i,
                                                                  batch_size=batch_size,
                                                                  target_layer=target_layer,
                                                                  len_prompt=len_inputs),
                                                              ) for layer in range(n_layers)]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            generation_toks = model.generate(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                generation_config=generation_config,
            )
            first_gen_toks = generation_toks[:, inputs.input_ids.shape[-1]]
            first_gen_str = tokenizer.batch_decode(first_gen_toks)
            first_gen_toks_all[i:i+batch_size] = first_gen_toks
            first_gen_str_all.append(first_gen_str)

            generation_toks = generation_toks[:, inputs.input_ids.shape[-1]:]
            for generation_idx, generation in enumerate(generation_toks):
                if labels is not None:
                    completions.append({
                        'prompt': dataset[i + generation_idx],
                        'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                        'label': labels[i + generation_idx],
                        'ID': i + generation_idx,
                        'category': categories[i + generation_idx],

                    })
                else:
                    completions.append({
                        'prompt': dataset[i + generation_idx],
                        'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                        'ID': i + generation_idx
                    })
    first_gen_str_all = [x for xs in first_gen_str_all for x in xs]
    return activations, completions, first_gen_toks_all, first_gen_str_all


def evaluate_jailbreak_completions(cfg, contrastive_label, dataset_name, eval_methodologies, few_shot=None):
    """Evaluate completions and save results for a dataset."""
    # with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
    with open(f'{cfg.artifact_path()}' + os.sep + 'completions' + os.sep + f'{dataset_name}' +
               '_completions_' + contrastive_label +'.json',
               "r") as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}' + os.sep + 'completions' + os.sep + f'{dataset_name}' +
               '_evaluations_' + contrastive_label +'.json', "w") as f:
        json.dump(evaluation, f, indent=4)


def generate_with_intervention_contrastive_activations_pca(cfg,
                                                           model_base,
                                                           dataset,
                                                           activations_positive,
                                                           activations_negative,
                                                           mean_diff=None,
                                                           labels_ori=None,
                                                           labels_int=None,
                                                           save_activations=False,
                                                           contrastive_label=["honest", "lying"],
                                                           categories=None):

    intervention = cfg.intervention
    source_layer = cfg.source_layer
    target_layer_s = cfg.target_layer_s
    target_layer_e = cfg.target_layer_e
    artifact_dir = cfg.artifact_path()
    model_name = cfg.model_alias
    data_category = cfg.data_category
    n_layers = model_base.model.config.num_hidden_layers
    if 'honest' in contrastive_label:
        tokenize_fn = model_base.tokenize_statements_fn
    elif 'HHH' in contrastive_label:
        tokenize_fn = model_base.tokenize_instructions_fn
    true_token_id = model_base.true_token_id
    false_token_id = model_base.false_token_id

    # 1. Generation with Intervention
    intervention_activations_positive, intervention_completions_positive, first_gen_toks_honest, first_gen_str_honest = get_intervention_activations_and_generation(
        cfg, model_base, dataset,
        tokenize_fn,
        mean_diff=mean_diff,
        positions=[-1],
        max_new_tokens=64,
        system_type=contrastive_label[0],
        target_layer_s=target_layer_s,
        target_layer_e=target_layer_e,
        labels=labels_int,
        categories=categories)
    intervention_activations_negative, intervention_completions_negative, first_gen_toks_lying, first_gen_str_lying = get_intervention_activations_and_generation(
        cfg, model_base, dataset,
        tokenize_fn,
        mean_diff=mean_diff,
        positions=[-1],
        max_new_tokens=64,
        system_type=contrastive_label[-1],
        target_layer_s=target_layer_s,
        target_layer_e=target_layer_e,
        labels=labels_int,
        categories=categories)

    # 2. save completions and activations
    if not os.path.exists(os.path.join(cfg.artifact_path(), intervention,  'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), intervention,  'completions'))

    if 'honest' in contrastive_label:
        with open(
                artifact_dir + os.sep + intervention + os.sep + 'completions' + os.sep 
                + f'{data_category}_{intervention}_completions_' +
                contrastive_label[0] + f'_layer_{source_layer}_{target_layer_s}_{target_layer_e}.json',
                "w") as f:
            json.dump(intervention_completions_positive, f, indent=4)
        with open(
                artifact_dir + os.sep + intervention + os.sep + 'completions' + os.sep 
                + f'{data_category}_{intervention}_completions_' + 
                contrastive_label[-1] + f'_layer_{source_layer}_{target_layer_s}_{target_layer_e}.json',
                "w") as f:
            json.dump(intervention_completions_negative, f, indent=4)
            
    elif 'HHH' in contrastive_label:
        # HHH persona
        # data = harmful
        with open(artifact_dir + os.sep + intervention + os.sep + 'completions' + os.sep +
                  'harmful_completions_' +
                  contrastive_label[0] + f'_layer_{source_layer}_{target_layer_s}_{target_layer_e}.json',
                  "w") as f:
            json.dump(intervention_completions_positive[:cfg.n_train], f, indent=4)
        # data= harmless
        with open(artifact_dir + os.sep + intervention + os.sep + 'completions' + os.sep +
                  'harmless_completions_' +
                  contrastive_label[0] + f'_layer_{source_layer}_{target_layer_s}_{target_layer_e}.json',
                  "w") as f:
            json.dump(intervention_completions_positive[cfg.n_train:], f, indent=4)

        # jailbreak persona
        # data= harmful
        with open(artifact_dir + os.sep + intervention + os.sep + 'completions' + os.sep +
                  'harmful_completions_' +
                  contrastive_label[1] + f'_layer_{source_layer}_{target_layer_s}_{target_layer_e}.json',
                  "w") as f:
            json.dump(intervention_completions_negative[:cfg.n_train], f, indent=4)
        # data= harmless
        with open(artifact_dir + os.sep + intervention + os.sep + 'completions' + os.sep +
                  'harmless_completions_' +
                  contrastive_label[1] + f'_layer_{source_layer}_{target_layer_s}_{target_layer_e}.json',
                  "w") as f:
            json.dump(intervention_completions_negative[cfg.n_train:], f, indent=4)

    # save activations
    if save_activations:
        if not os.path.exists(os.path.join(cfg.artifact_path(), intervention, 'activations')):
            os.makedirs(os.path.join(cfg.artifact_path(), intervention, 'activations'))

        activations = {
            "activations_positive": intervention_activations_positive,
            "activations_negative": intervention_activations_negative,
        }
        with open(artifact_dir + os.sep + intervention + os.sep + 'activations' + os.sep +
                  os.sep + model_name + '_' + f'{data_category}' +
                  '_activation_pca_' + intervention + '_' + str(source_layer) + '_' + str(target_layer_s) +
                  '_' + str(target_layer_e) + '_' + contrastive_label[1] + '.pkl', "wb") as f:
            pickle.dump(activations, f)

    # 3. pca with and without intervention, plot and save pca plots
    if "skip_connection" in intervention:
        plot_original = False
        plot_intervention = True
    else:
        plot_original = True
        plot_intervention = True
    contrastive_intervention_label = [contrastive_label[0], contrastive_label[1],
                                      contrastive_label[0] + "_intervention", contrastive_label[1] + "_intervention"]
    fig = plot_contrastive_activation_intervention_pca(activations_positive,
                                                       activations_negative,
                                                       intervention_activations_positive,
                                                       intervention_activations_negative,
                                                       n_layers,
                                                       contrastive_intervention_label,
                                                       labels_ori,
                                                       labels_int,
                                                       plot_original=plot_original,
                                                       plot_intervention=plot_intervention,
                                                       )
    fig.write_html(artifact_dir + os.sep + intervention + os.sep + data_category + '_' + intervention +
                   '_pca_layer_' + str(source_layer) + '_' + str(target_layer_s) + '_' + str(target_layer_e) +
                    '_' + contrastive_label[1] + '.html')
    pio.write_image(fig, artifact_dir + os.sep + intervention + os.sep + data_category + '_' + intervention +
                   '_pca_layer_' + str(source_layer) + '_' + str(target_layer_s) + '_' + str(target_layer_e) +
                     '_' + contrastive_label[1] + '.png',
                    scale=6)

    # 4. get performance
    # todo: get it compatible with jailbreak
    if 'honest' in contrastive_label:
        model_performance, fig = get_performance_stats(cfg, first_gen_toks_honest, first_gen_str_honest,
                                                       first_gen_toks_lying, first_gen_str_lying,
                                                       labels_int,
                                                       true_token_id, false_token_id
                                                       )
        fig.write_html(artifact_dir + os.sep + intervention + os.sep + f'{data_category}_{intervention}_'
                       + 'model_performance_layer_' + str(source_layer) + '_' + str(target_layer_s) + '_' + str(
                         target_layer_e) + '_accuracy' +
                       '.html')
        with open(artifact_dir + os.sep + intervention + os.sep + f'{data_category}' +
                  '_model_performance_' + intervention + '_layer_' + str(source_layer) + '_' + str(target_layer_s) +
                  '_' + str(target_layer_e) + '.pkl', "wb") as f:
            pickle.dump(model_performance, f)

    elif "HHH" in contrastive_label:
        dataset_name = ['harmful', 'harmless']
        save_path = artifact_dir + os.sep + intervention + os.sep
        evaluate_completions_and_save_results_for_dataset(cfg, dataset_name,
                                                          cfg.jailbreak_eval_methodologies,
                                                          contrastive_label,
                                                          save_path,
                                                          source_layer=source_layer,
                                                          target_layer_s=target_layer_s,
                                                          target_layer_e=target_layer_e,
                                                          few_shot=None)

    # 4. Get stage statistics with intervention
    # stage_stats_intervention = get_state_quantification(cfg, intervention_activations_positive,
    #                                                     intervention_activations_negative,
    #                                                     labels_int,
    #                                                     save_plot=False)
    # intervention_results = {
    #     'intervention_activations_positive': intervention_activations_positive,
    #     'intervention_activations_negative': intervention_activations_negative,
    #     # 'model_performance': model_performance,
    #     'stage_stats_intervention': stage_stats_intervention
    #
    # }
    # return intervention_results


# def extraction_intervention_and_plot_pca(cfg,model_base: ModelBase, harmful_instructions, harmless_instructions):
#     artifact_dir = cfg.artifact_path()
#     if not os.path.exists(artifact_dir):
#         os.makedirs(artifact_dir)
#
#     # 1. extract activations
#     activations_positive,activations_negative = generate_activations_and_plot_pca(cfg,model_base, harmful_instructions, harmless_instructions)
#
#     # 2. get steering vector = get mean difference of the source layer
#     mean_activation_harmful = activations_positive.mean(dim=0)
#     mean_activation_harmless = activations_negative.mean(dim=0)
#     mean_diff = mean_activation_harmful-mean_activation_harmless
#
#
#     # 3. refusal_intervention_and_plot_pca
#     intervention_activations_positive,intervention_activations_negative = refusal_intervention_and_plot_pca(cfg, model_base,
#                                                                                                    harmful_instructions,
#                                                                                                    harmless_instructions,
#                                                                                                    mean_diff)
#
#     # 4. pca with and without intervention, plot and save pca
#     intervention = cfg.intervention
#     source_layer = cfg.source_layer
#     target_layer = cfg.target_layer
#     model_name = cfg.model_alias
#     n_layers = model_base.model.config.num_hidden_layers
#     fig = plot_contrastive_activation_intervention_pca(activations_positive, activations_negative,
#                                                        intervention_activations_positive, intervention_activations_negative,
#                                                        n_layers)
#     fig.write_html(artifact_dir + os.sep + model_name + '_' + 'refusal_generation_activation_'
#                    +intervention+'_pca_layer_'
#                    +str(source_layer)+'_'+ str(target_layer)+'.html')