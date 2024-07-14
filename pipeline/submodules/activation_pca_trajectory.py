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

from pipeline.utils.hook_utils import get_and_cache_direction_ablation_input_pre_hook
from pipeline.utils.hook_utils import get_and_cache_diff_addition_input_pre_hook
from pipeline.utils.hook_utils import get_and_cache_direction_ablation_output_hook
from pipeline.utils.hook_utils import get_generation_cache_activation_trajectory_input_pre_hook
from pipeline.utils.hook_utils import get_activations_pre_hook, get_generation_cache_activation_input_pre_hook


def plot_contrastive_activation_pca_with_trajectory_one_layer(activations_honest, activations_lie,
                                                              activation_trajectory_honest, activation_trajectory_lie,
                                                              n_layers,
                                                              str_honest, str_lie,
                                                              contrastive_label=["honest", "lying", "trajectory_honest", "trajectory_lying"],
                                                              labels=None,
                                                              layer=79
                                                              ):

    n_contrastive_data = activations_lie.shape[0]
    n_trajectory_honest = len(str_honest)
    n_trajectory_lie = len(str_lie)
    # only take the part of the answer that is actually generated (otherwise the 0 will mess up pca and produce nan values)
    activation_trajectory_lie = activation_trajectory_lie[:, :n_trajectory_lie, :, :]
    activation_trajectory_honest = activation_trajectory_honest[:, :n_trajectory_honest, :, :]
    # reshape the activation trajectory to to the same format as the activations
    activation_trajectory_lie = einops.rearrange(activation_trajectory_lie,
                                                 '1 seq n_layers d_model -> seq n_layers d_model')
    labels_trajectory_lie = [2 + 0 * i for i in range(activation_trajectory_lie.shape[0])]
    activation_trajectory_honest = einops.rearrange(activation_trajectory_honest,
                                                    '1 seq n_layers d_model -> seq n_layers d_model')
    labels_trajectory_honest = [3 + 0 * i for i in range(activation_trajectory_honest.shape[0])]
    # concatenate all
    # activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_honest,
    #                                                                           activations_lie,
    #                                                                           activation_trajectory_honest,
    #                                                                           activation_trajectory_lie), dim=0)

    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_honest,
                                                                              activations_lie), dim=0)
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
        label_text = np.append(label_text, f'{contrastive_label[0]}_{labels_tf[ii]}_{ii}')
    for ii in range(n_contrastive_data):
        label_text = np.append(label_text, f'{contrastive_label[1]}_{labels_tf[ii]}_{ii}')
    for ii in range(n_trajectory_honest):
        label_text = np.append(label_text, f'{contrastive_label[2]}_{str_honest[ii]}_{ii}')
    for ii in range(n_trajectory_lie):
        label_text = np.append(label_text, f'{contrastive_label[3]}_{str_lie[ii]}_{ii}')

    pca = PCA(n_components=3)

    fig = make_subplots(rows=1, cols=1)
    activations_pca_trajectory = pca.fit_transform(activations_all_trajectory[:, layer, :].cpu())
    activations_pca = pca.transform(activations_all[:, layer, :].cpu())
    activations_pca_all = np.concatenate((activations_pca, activations_pca_trajectory), axis=0)

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
    )
    # plot lying data
    fig.add_trace(
        go.Scatter(x=df['pca0'][n_contrastive_data:n_contrastive_data * 2],
                   y=df['pca1'][n_contrastive_data:n_contrastive_data * 2],
                   mode="markers",
                   showlegend=False,
                   marker=dict(
                       symbol="circle",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                       color=df['label'][n_contrastive_data:n_contrastive_data * 2],
                   ),
                   text=df['label_text'][n_contrastive_data:n_contrastive_data * 2]),
    )

    # plot honest trajectory
    fig.add_trace(
        go.Scatter(x=df['pca0'][n_contrastive_data * 2:n_contrastive_data * 2 + n_trajectory_honest],
                   y=df['pca1'][n_contrastive_data * 2:n_contrastive_data * 2 + n_trajectory_honest],
                   mode="markers",
                   showlegend=False,
                   marker=dict(
                       symbol="square",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                       color="green",
                   ),
                   text=df['label_text'][n_contrastive_data * 2:n_contrastive_data * 2 + n_trajectory_honest]),
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
    )
    fig.update_layout(height=500, width=700)
    fig.show()
    fig.write_html('honest_lying_pca.html')


