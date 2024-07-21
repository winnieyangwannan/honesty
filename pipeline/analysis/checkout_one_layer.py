import os
import argparse
from pipeline.honesty_config_generation_skip_connection import Config
from pipeline.model_utils.model_factory import construct_model_base
import pickle
import csv
import math
from tqdm import tqdm
from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from plotly.figure_factory import create_quiver
import plotly.figure_factory as ff
import plotly.io as pio


from sklearn.decomposition import PCA
import numpy as np
import torch

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor


from scipy.spatial.distance import cdist
from scipy import stats


def plot_one_layer_3d(activations_honest, activations_lying,
                      labels,
                      prompt_label=["honest", "lying"],
                      layer=16):

    n_samples = activations_honest.shape[0]
    if labels is not None:
        labels_all = labels + labels
        # true or false label
        labels_tf = []
        for ll in labels:
            if ll == 0:
                labels_tf.append('false')
            elif ll == 1:
                labels_tf.append('true')

    label_text = []
    for ii in range(n_samples):
        label_text = np.append(label_text, f'{prompt_label[0]}_{labels_tf[ii]}_{ii}')
    for ii in range(n_samples):
        label_text = np.append(label_text, f'{prompt_label[1]}_{labels_tf[ii]}_{ii}')

    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_honest,
                                                                              activations_lying), dim=0)

    pca = PCA(n_components=3)
    activations_pca_all = pca.fit_transform(activations_all[:, layer, :].cpu())
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=f"layer {layer}",
                        specs=[[{'type': 'scene'}]])
    df = {}
    df['label'] = labels_all
    df['pca0'] = activations_pca_all[:, 0]
    df['pca1'] = activations_pca_all[:, 1]
    df['pca2'] = activations_pca_all[:, 1]
    df['label_text'] = label_text
    fig.add_trace(
        go.Scatter3d(x=df['pca0'][:n_samples],
                     y=df['pca1'][:n_samples],
                     z=df['pca2'][:n_samples],
                     mode="markers",
                     showlegend=False,
                     marker=dict(
                         symbol="cross",
                         size=8,
                         line=dict(width=1, color="DarkSlateGrey"),
                         color=df['label'][:n_samples]
                     ),
                     text=df['label_text'][:n_samples]),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter3d(x=df['pca0'][n_samples:n_samples * 2],
                     y=df['pca1'][n_samples:n_samples * 2],
                     z=df['pca2'][n_samples:n_samples * 2],
                     mode="markers",
                     showlegend=False,
                     marker=dict(
                         symbol="circle",
                         size=5,
                         line=dict(width=1, color="DarkSlateGrey"),
                         color=df['label'][n_samples:n_samples * 2],
                     ),
                     text=df['label_text'][n_samples:n_samples * 2]),
        row=1, col=1,
    )

    fig.update_layout(height=1000, width=1000)
    fig.write_html(f'pca_layer_3d_layer_{layer}.html')
    return fig


def plot_one_layer_with_centeroid_and_vector(activations_honest, activations_lying,
                                             centeroid_honest_true, centeroid_honest_false,
                                             centeroid_lying_true, centeroid_lying_false,
                                             centeroid_vector_honest, centeroid_vector_lying,
                                             labels,
                                             save_path,
                                             prompt_label=["honest", "lying"],
                                             layer=16):

    n_samples = activations_honest.shape[0]
    if labels is not None:
        labels_all = labels + labels
        # true or false label
        labels_tf = []
        for ll in labels:
            if ll == 0:
                labels_tf.append('false')
            elif ll == 1:
                labels_tf.append('true')

    label_text = []
    for ii in range(n_samples):
        label_text = np.append(label_text, f'{prompt_label[0]}_{labels_tf[ii]}_{ii}')
    for ii in range(n_samples):
        label_text = np.append(label_text, f'{prompt_label[1]}_{labels_tf[ii]}_{ii}')

    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_honest,
                                                                              activations_lying), dim=0)

    pca = PCA(n_components=3)
    activations_pca_all = pca.fit_transform(activations_all[:, layer, :].cpu())

    df = {}
    df['label'] = labels_all
    df['pca0'] = activations_pca_all[:, 0]
    df['pca1'] = activations_pca_all[:, 1]
    df['pca2'] = activations_pca_all[:, 1]
    df['label_text'] = label_text

    # plot the centeroid vector
    # x, y = np.meshgrid(np.arange(-10, 10, .2),
    #                    np.arange(-10, 10, .2))
    # x y of quiver is the origin of the vector, u v is the end point of the vector
    fig = ff.create_quiver(x=[centeroid_honest_false[0], centeroid_lying_false[0]],
                           y=[centeroid_honest_false[1], centeroid_lying_false[1]],
                           u=[centeroid_vector_honest[0], centeroid_vector_lying[0]],
                           v=[centeroid_vector_honest[1], centeroid_vector_lying[1]],
                           line=dict(width=3, color='black'),
                           scale=1)

    fig.add_trace(
        go.Scatter(x=df['pca0'][:n_samples],
                   y=df['pca1'][:n_samples],
                   mode="markers",
                   showlegend=False,
                   marker=dict(
                         symbol="cross",
                         size=8,
                         line=dict(width=1, color="DarkSlateGrey"),
                         color=df['label'][:n_samples]
                   ),
                   text=df['label_text'][:n_samples]),
    )

    fig.add_trace(
        go.Scatter(x=df['pca0'][n_samples:n_samples * 2],
                   y=df['pca1'][n_samples:n_samples * 2],
                   mode="markers",
                   showlegend=False,
                   marker=dict(
                         symbol="circle",
                         size=5,
                         line=dict(width=1, color="DarkSlateGrey"),
                         color=df['label'][n_samples:n_samples * 2],
                   ),
                   text=df['label_text'][n_samples:n_samples * 2]),
    )
    # plot centroid: centroid honest, true
    fig.add_trace(go.Scatter(
        x=[centeroid_honest_true[0]],
        y=[centeroid_honest_true[1]],
        marker=dict(
            symbol='cross',
            size=10,
            color='orange'
        ),
        name='hoenst_true_centeroid'
    ))
    # plot centroid: centroid honest, false
    fig.add_trace(go.Scatter(
        x=[centeroid_honest_false[0]],
        y=[centeroid_honest_false[1]],
        marker=dict(
            symbol='cross',
            size=10,
            color='blue'
        ),
        name='hoenst_false_centeroid'
    ))
    # plot centroid: centroid lying, true
    fig.add_trace(go.Scatter(
        x=[centeroid_lying_true[0]],
        y=[centeroid_lying_true[1]],
        marker=dict(
            symbol='circle',
            size=10,
            color='orange'
        ),
        name='lying_true_centeroid',
    ))
    # plot centroid: centroid lying, false
    fig.add_trace(go.Scatter(
        x=[centeroid_lying_false[0]],
        y=[centeroid_lying_false[1]],
        marker=dict(
            symbol='circle',
            size=10,
            color='blue'
        ),
        name='lying_false_centeroid',
    ))
    fig.update_layout(height=800, width=800)
    fig.write_html(save_path + os.sep + 'pca_layer.html')
    pio.write_image(save_path + os.sep + 'pca_layer.png')
    pio.write_image(save_path + os.sep + 'pca_layer.pdf')

    return fig
