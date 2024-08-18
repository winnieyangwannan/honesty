import os
import argparse
from pipeline.jailbreak_config_generation import Config
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


def plot_contrastive_activation_pca_layer_jailbreaks(cfg,
                                                     activations_all,
                                                     contrastive_labels_all,
                                                     contrastive_type,
                                                     prompt_labels_all,
                                                     prompt_type,
                                                     ):
    print("plot")
    n_layers = activations_all.shape[1]
    layers = np.arange(n_layers)
    n_contrastive_data = cfg.n_train
    n_contrastive_groups = int(len(activations_all)/n_contrastive_data/2)
    colors = ['yellow', 'red', 'blue']

    label_text = []
    for ii in range(len(activations_all)):
        if int(prompt_labels_all[ii]) == 0:
             label_text = np.append(label_text, f'{contrastive_labels_all[ii]}_{prompt_type[0]}')
        if int(prompt_labels_all[ii]) == 1:
             label_text = np.append(label_text, f'{contrastive_labels_all[ii]}_{prompt_type[1]}')

    cols = 4
    rows = math.ceil(n_layers/cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"layer {n}" for n in (layers)])

    pca = PCA(n_components=3)
    for row in range(rows):
        for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
            # print(f'layer{layer}')
            if layer < n_layers:
                # print(f'layer{layer}')
                activations_pca = pca.fit_transform(activations_all[:, layer, :].cpu())
                df = {}
                df['label'] = contrastive_labels_all
                df['pca0'] = activations_pca[:, 0]
                df['pca1'] = activations_pca[:, 1]
                df['pca2'] = activations_pca[:, 2]
                df['label_text'] = label_text


                for ii in range(n_contrastive_groups):
                    fig.add_trace(
                        go.Scatter(x=df['pca0'][ii*n_contrastive_data*2:ii*n_contrastive_data*2+n_contrastive_data],
                                   y=df['pca1'][ii*n_contrastive_data*2:ii*n_contrastive_data*2+n_contrastive_data],
                                     # z=df['pca2'][:n_contrastive_data],
                                     mode="markers",
                                     showlegend=False,
                                     marker=dict(
                                     symbol="star",
                                     size=4,
                                     line=dict(width=1, color="DarkSlateGrey"),
                                     color=colors[ii]
                                     ),
                                   text=df['label_text'][ii*n_contrastive_data*2:ii*n_contrastive_data*2+n_contrastive_data]),
                        row=row+1, col=ll+1)

                    fig.add_trace(
                        go.Scatter(x=df['pca0'][ii*n_contrastive_data*2+n_contrastive_data:(ii+1)*n_contrastive_data*2],
                                   y=df['pca1'][ii*n_contrastive_data*2+n_contrastive_data:(ii+1)*n_contrastive_data*2],
                                   # z=df['pca2'][:n_contrastive_data],
                                   mode="markers",
                                   showlegend=False,
                                   marker=dict(
                                       symbol="circle",
                                       size=4,
                                       line=dict(width=1, color="DarkSlateGrey"),
                                       color=colors[ii],
                                   ),
                                   text=df['label_text'][ii*n_contrastive_data*2+n_contrastive_data:(ii+1)*n_contrastive_data*2]),
                        row=row + 1, col=ll + 1)
    # legend
    for ii in range(n_contrastive_groups):
        fig.add_trace(
            go.Scatter(x=[None],
                     y=[None],
                         # z=[None],
                         mode='markers',
                         marker=dict(
                           symbol="star",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color=colors[0],
                         ),
                         name=f'{label_text[ii*n_contrastive_data*2]}',
                         marker_color=colors[ii],
                         ),
            row=row + 1, col=ll + 1,
        )

    fig.update_layout(height=1600, width=1000)

    fig.show()

    return fig


def plot_contrastive_activation_pca_one_layer_jailbreaks(cfg,
                                                         activations_all,
                                                         contrastive_labels_all,
                                                         contrastive_type,
                                                         prompt_labels_all,
                                                         prompt_type,
                                                         layer_plot=10
                                                         ):
    print("plot")
    n_layers = activations_all.shape[1]
    # layers = np.arange(n_layers)
    n_contrastive_data = cfg.n_train
    n_contrastive_groups = int(len(activations_all)/n_contrastive_data/2)
    colors = ['yellow', 'red', 'blue']

    label_text = []
    for ii in range(len(activations_all)):
        if int(prompt_labels_all[ii]) == 0:
             label_text = np.append(label_text, f'{contrastive_labels_all[ii]}_{prompt_type[0]}')
        if int(prompt_labels_all[ii]) == 1:
             label_text = np.append(label_text, f'{contrastive_labels_all[ii]}_{prompt_type[1]}')

    cols = 4
    # rows = math.ceil(n_layers/cols)
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=[f""])

    pca = PCA(n_components=3)

    # print(f'layer{layer}')
    activations_pca = pca.fit_transform(activations_all[:, layer_plot, :].cpu())
    df = {}
    df['label'] = contrastive_labels_all
    df['pca0'] = activations_pca[:, 0]
    df['pca1'] = activations_pca[:, 1]
    df['pca2'] = activations_pca[:, 2]
    df['label_text'] = label_text

    for ii in range(n_contrastive_groups):
        fig.add_trace(
            go.Scatter(x=df['pca0'][ii*n_contrastive_data*2:ii*n_contrastive_data*2+n_contrastive_data],
                       y=df['pca1'][ii*n_contrastive_data*2:ii*n_contrastive_data*2+n_contrastive_data],
                         # z=df['pca2'][:n_contrastive_data],
                         mode="markers",
                         showlegend=False,
                         marker=dict(
                         symbol="star",
                         size=4,
                         line=dict(width=1, color="DarkSlateGrey"),
                         color=colors[ii]
                         ),
                       text=df['label_text'][ii*n_contrastive_data*2:ii*n_contrastive_data*2+n_contrastive_data]),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=df['pca0'][ii*n_contrastive_data*2+n_contrastive_data:(ii+1)*n_contrastive_data*2],
                       y=df['pca1'][ii*n_contrastive_data*2+n_contrastive_data:(ii+1)*n_contrastive_data*2],
                       # z=df['pca2'][:n_contrastive_data],
                       mode="markers",
                       showlegend=False,
                       marker=dict(
                           symbol="circle",
                           size=4,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color=colors[ii],
                       ),
                       text=df['label_text'][ii*n_contrastive_data*2+n_contrastive_data:(ii+1)*n_contrastive_data*2]),
            row=1, col=1)

    # legend
    for ii in range(n_contrastive_groups):
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       # z=[None],
                       mode='markers',
                       marker=dict(
                           symbol="star",
                           size=5,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color=colors[0],
                         ),
                       name=f'{label_text[ii*n_contrastive_data*2]}',
                       marker_color=colors[ii],
                       ),
            row=1, col=1,
        )

    fig.update_layout(height=400, width=500,
                      title=dict(text=f"Layer {layer_plot}", font=dict(size=30), automargin=True, yref='paper')
                      )
    fig.show()

    return fig


def plot_contrastive_activation_pca_one_layer_jailbreaks_3d(cfg,
                                                            activations_all,
                                                            contrastive_labels_all,
                                                            contrastive_type,
                                                            prompt_labels_all,
                                                            prompt_type,
                                                             layer_plot=10
                                                             ):
    print("plot")
    n_layers = activations_all.shape[1]
    # layers = np.arange(n_layers)
    n_contrastive_data = cfg.n_train
    n_contrastive_groups = int(len(activations_all)/n_contrastive_data/2)
    colors = ['yellow', 'red', 'blue']

    label_text = []
    for ii in range(len(activations_all)):
        if int(prompt_labels_all[ii]) == 0:
             label_text = np.append(label_text, f'{contrastive_labels_all[ii]}_{prompt_type[0]}')
        if int(prompt_labels_all[ii]) == 1:
             label_text = np.append(label_text, f'{contrastive_labels_all[ii]}_{prompt_type[1]}')

    cols = 4
    # rows = math.ceil(n_layers/cols)
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=[f""],
                        specs=[[{'type': 'scene'}]])

    pca = PCA(n_components=3)

    # print(f'layer{layer}')
    activations_pca = pca.fit_transform(activations_all[:, layer_plot, :].cpu())
    df = {}
    df['label'] = contrastive_labels_all
    df['pca0'] = activations_pca[:, 0]
    df['pca1'] = activations_pca[:, 1]
    df['pca2'] = activations_pca[:, 2]
    df['label_text'] = label_text

    for ii in range(n_contrastive_groups):
        fig.add_trace(

            go.Scatter3d(x=df['pca0'][ii*n_contrastive_data*2:ii*n_contrastive_data*2+n_contrastive_data],
                         y=df['pca1'][ii*n_contrastive_data*2:ii*n_contrastive_data*2+n_contrastive_data],
                         z=df['pca1'][ii * n_contrastive_data * 2:ii * n_contrastive_data * 2 + n_contrastive_data],
                         # z=df['pca2'][:n_contrastive_data],
                         mode="markers",
                         showlegend=False,
                         marker=dict(
                         symbol="cross",
                         size=4,
                         line=dict(width=1, color="DarkSlateGrey"),
                         color=colors[ii]
                         ),
                        text=df['label_text'][ii*n_contrastive_data*2:ii*n_contrastive_data*2+n_contrastive_data]),
            row=1, col=1)

        fig.add_trace(
            go.Scatter3d(x=df['pca0'][ii*n_contrastive_data*2+n_contrastive_data:(ii+1)*n_contrastive_data*2],
                         y=df['pca1'][ii*n_contrastive_data*2+n_contrastive_data:(ii+1)*n_contrastive_data*2],
                         z=df['pca1'][ii * n_contrastive_data * 2 + n_contrastive_data:(ii + 1) * n_contrastive_data * 2],
                         # z=df['pca2'][:n_contrastive_data],
                         mode="markers",
                         showlegend=False,
                         marker=dict(
                           symbol="circle",
                           size=4,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color=colors[ii],
                         ),
                         text=df['label_text'][ii*n_contrastive_data*2+n_contrastive_data:(ii+1)*n_contrastive_data*2]),
            row=1, col=1)

    # legend
    for ii in range(n_contrastive_groups):
        fig.add_trace(
            go.Scatter3d(x=[None],
                       y=[None],
                       z=[None],
                       mode='markers',
                       marker=dict(
                           symbol="cross",
                           size=4,
                           line=dict(width=1, color="DarkSlateGrey"),
                           color=colors[0],
                         ),
                       name=f'{label_text[ii*n_contrastive_data*2]}',
                       marker_color=colors[ii],
                       ),
            row=1, col=1,
        )

    fig.update_layout(height=400, width=500,
                      title=dict(text=f"Layer {layer_plot}", font=dict(size=30), automargin=True, yref='paper')
                      )
    fig.show()

    return fig