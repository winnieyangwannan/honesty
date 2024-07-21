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

from pipeline.utils.hook_utils import get_and_cache_direction_ablation_input_pre_hook
from pipeline.utils.hook_utils import get_and_cache_diff_addition_input_pre_hook, get_and_cache_diff_addition_output_hook
from pipeline.utils.hook_utils import get_and_cache_direction_ablation_output_hook
from pipeline.utils.hook_utils import get_and_cache_skip_connection_input_pre_hook, get_and_cache_skip_connection_hook


def plot_contrastive_activation_intervention_pca(activations_honest,
                                                 activations_lie,
                                                 intervention_activations_honest,
                                                 intervention_activations_lie,
                                                 n_layers,
                                                 contrastive_label,
                                                 labels=None,
                                                 plot_original=False,
                                                 plot_intervention=True,
                                                 ):
    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_honest,
                                                                              activations_lie),
                                                                              dim=0)
    activations_all_intervention = torch.cat((
                                              intervention_activations_honest,
                                              intervention_activations_lie),
                                              dim=0)
    pca = PCA(n_components=3)

    label_text = []
    label_plot = []
    n_data = activations_honest.shape[0]
    # true or false label
    labels_tf = []
    for ll in labels:
        if ll == 0:
            labels_tf.append('false')
        elif ll == 1:
            labels_tf.append('true')

    for ii in range(n_data):
        label_text = np.append(label_text, f'{contrastive_label[0]}_{labels_tf[ii]}:{ii}')
        # label_plot.append(0)
    for ii in range(n_data):
        label_text = np.append(label_text, f'{contrastive_label[1]}_{labels_tf[ii]}:{ii}')
        # label_plot.append(1)
    for ii in range(n_data):
        label_text = np.append(label_text, f'{contrastive_label[2]}_{labels_tf[ii]}:{ii}')
        # label_plot.append(2)
    for ii in range(n_data):
        label_text = np.append(label_text, f'{contrastive_label[3]}_{labels_tf[ii]}:{ii}')
        # label_plot.append(3)
    label_plot = labels + labels + labels + labels
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
                # activations_pca = pca.fit_transform(activations_all[:, layer, :].cpu())
                # activations_pca_intervention = pca.transform(activations_all_intervention[:, layer, :].cpu())
                # activation_pca_all = np.concatenate((activations_pca, activations_pca_intervention), axis=0)

                # project to the intervened honest and lying space
                activations_pca_intervention = pca.fit_transform(activations_all_intervention[:, layer, :].cpu())
                activations_pca = pca.transform(activations_all[:, layer, :].cpu())
                activation_pca_all = np.concatenate((activations_pca, activations_pca_intervention), axis=0)

                df = {}
                df['label'] = label_plot
                df['pca0'] = activation_pca_all[:, 0]
                df['pca1'] = activation_pca_all[:, 1]
                df['label_text'] = label_text

                if plot_original:
                    fig.add_trace(
                        go.Scatter(x=df['pca0'][:n_data],
                                   y=df['pca1'][:n_data],
                                   mode='markers',
                                   showlegend=False,
                                   marker=dict(
                                       symbol="star",
                                       size=8,
                                       line=dict(width=1, color="DarkSlateGrey"),
                                       color=df['label'][:n_data],
                                       colorscale='PiYG'
                                   ),
                                   text=df['label_text'][:n_data]),
                        row=row + 1, col=ll + 1,
                    )

                    fig.add_trace(
                        go.Scatter(x=df['pca0'][n_data:n_data*2],
                                   y=df['pca1'][n_data:n_data*2],
                                   mode='markers',
                                   showlegend=False,
                                   marker=dict(
                                       symbol="circle",
                                       size=8,
                                       line=dict(width=1, color="DarkSlateGrey"),
                                       color=df['label'][n_data:n_data*2],
                                       colorscale='PiYG'
                                   ),
                                   text=df['label_text'][n_data:n_data*2]),
                        row=row + 1, col=ll + 1,
                    )
                if plot_intervention:
                    fig.add_trace(
                        go.Scatter(x=df['pca0'][n_data*2:n_data*3],
                                   y=df['pca1'][n_data*2:n_data*3],
                                   mode='markers',
                                   showlegend=False,
                                   marker=dict(
                                       symbol="triangle-up",
                                       size=8,
                                       line=dict(width=1, color="DarkSlateGrey"),
                                       color=df['label'][n_data*2:n_data*3],
                                   ),
                                   text=df['label_text'][n_data*2:n_data*3]),
                        row=row + 1, col=ll + 1,
                    )
                    fig.add_trace(
                        go.Scatter(x=df['pca0'][-n_data:],
                                   y=df['pca1'][-n_data:],
                                   mode='markers',
                                   showlegend=False,
                                   marker=dict(
                                       symbol="square",
                                       size=8,
                                       line=dict(width=1, color="DarkSlateGrey"),
                                       color=df['label'][-n_data:],
                                   ),
                                   text=df['label_text'][-n_data:]),
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
                       name=f'honest_true',
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
                       name=f'honest_false',
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
                       name=f'lying_true',
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
                       name=f'lying_false',
                       ),
        )
    if plot_intervention:
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="triangle-up",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color='yellow'
                       ),
                       name=f'honest_intervention_true',
                       ),
        )
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="triangle-up",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color='blue'
                       ),
                       name=f'honest_intervention_false',
                       ),
        )
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="square",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color='yellow'
                       ),
                       name=f'lying_intervention_true',
                       ),
        )
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(
                           symbol="square",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color='blue'

                       ),
                       name=f'lying_intervention_false',
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
                                                labels=None):
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
    elif type(target_layer_s) == int and target_layer_e is None:
        target_layer = [np.arange(n_layers)[target_layer_s]]
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

        if 'direction ablation' in intervention:
            if "mlp" in intervention:
                fwd_pre_hooks = [(block_modules[layer].mlp,
                                  get_and_cache_direction_ablation_input_pre_hook(
                                      mean_diff=mean_diff,
                                      cache=activations,
                                      layer=layer,
                                      positions=positions,
                                      batch_id=i,
                                      batch_size=batch_size,
                                      target_layer=target_layer,
                                      len_prompt=len_inputs),
                                  ) for layer in range(n_layers)]
                fwd_hooks = [(block_modules[layer].mlp,
                              get_and_cache_direction_ablation_output_hook(
                                  mean_diff=mean_diff,
                                  layer=layer,
                                  positions=positions,
                                  batch_id=i,
                                  batch_size=batch_size,
                                  target_layer=target_layer,
                              ),
                              ) for layer in range(n_layers)]
            else:
                fwd_pre_hooks = [(block_modules[layer],
                                  get_and_cache_direction_ablation_input_pre_hook(
                                                           mean_diff=mean_diff,
                                                           cache=activations,
                                                           layer=layer,
                                                           positions=positions,
                                                           batch_id=i,
                                                           batch_size=batch_size,
                                                           target_layer=target_layer,
                                                           len_prompt=len_inputs),
                                                        ) for layer in range(n_layers)]
                fwd_hooks = [(block_modules[layer],
                              get_and_cache_direction_ablation_output_hook(
                                                       mean_diff=mean_diff,
                                                       layer=layer,
                                                       positions=positions,
                                                       batch_id=i,
                                                       batch_size=batch_size,
                                                       target_layer=target_layer,
                                                       ),
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
                        'ID': i + generation_idx

                    })
                else:
                    completions.append({
                        'prompt': dataset[i + generation_idx],
                        'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                        'ID': i + generation_idx
                    })
    first_gen_str_all = [x for xs in first_gen_str_all for x in xs]
    return activations, completions, first_gen_toks_all, first_gen_str_all

#
# def extraction_intervention_and_plot_pca(cfg,model_base: ModelBase, harmful_instructions, harmless_instructions):
#     artifact_dir = cfg.artifact_path()
#     if not os.path.exists(artifact_dir):
#         os.makedirs(artifact_dir)
#
#     # 1. extract activations
#     activations_honest,activations_lie = generate_activations_and_plot_pca(cfg,model_base, harmful_instructions, harmless_instructions)
#
#     # 2. get steering vector = get mean difference of the source layer
#     mean_activation_harmful = activations_honest.mean(dim=0)
#     mean_activation_harmless = activations_lie.mean(dim=0)
#     mean_diff = mean_activation_harmful-mean_activation_harmless
#
#
#     # 3. refusal_intervention_and_plot_pca
#     intervention_activations_honest,intervention_activations_lie = refusal_intervention_and_plot_pca(cfg, model_base,
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
#     fig = plot_contrastive_activation_intervention_pca(activations_honest, activations_lie,
#                                                        intervention_activations_honest, intervention_activations_lie,
#                                                        n_layers)
#     fig.write_html(artifact_dir + os.sep + model_name + '_' + 'refusal_generation_activation_'
#                    +intervention+'_pca_layer_'
#                    +str(source_layer)+'_'+ str(target_layer)+'.html')