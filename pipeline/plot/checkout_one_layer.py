import os
import argparse
from pipeline.honesty_config_generation_intervention import Config
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


def plot_contrastive_activation_pca_layer(activations_positive, activations_negative,
                                          contrastive_label,
                                          labels=None, prompt_label=['true', 'false'],
                                          layers=[0, 1]):

    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_positive,
                                                                              activations_negative), dim=0)

    n_contrastive_data = activations_negative.shape[0]
    n_layers = len(layers)
    if labels is not None:
        labels_all = labels + labels
        # true or false label
        labels_tf = []
        for ll in labels:
            if ll == 0:
                labels_tf.append(prompt_label[1])
            elif ll == 1:
                labels_tf.append(prompt_label[0])
    else:
        labels_all = np.zeros((n_contrastive_data*2,1))
        labels_tf = np.zeros((n_contrastive_data*2,1))

    label_text = []
    for ii in range(n_contrastive_data):
        label_text = np.append(label_text, f'{contrastive_label[0]}_{labels_tf[ii]}_{ii}')
    for ii in range(n_contrastive_data):
        label_text = np.append(label_text, f'{contrastive_label[1]}_{labels_tf[ii]}_{ii}')

    cols = 4
    rows = math.ceil(n_layers/cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"layer {n}" for n in (layers)])

    pca = PCA(n_components=3)
    for row in range(rows):
        for ll, layer in enumerate(layers):
            # print(f'layer{layer}')
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
                             name=contrastive_label[0],
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
            fig.add_trace(
                go.Scatter(x=df['pca0'][n_contrastive_data:],
                             y=df['pca1'][n_contrastive_data:],
                             # z=df['pca2'][n_contrastive_data:],
                             mode="markers",
                             name=contrastive_label[1],
                             showlegend=False,
                             marker=dict(
                               symbol="circle",
                               size=5,
                               line=dict(width=1, color="DarkSlateGrey"),
                               color=df['label'][n_contrastive_data:],
                             ),
                             text=df['label_text'][n_contrastive_data:]),
                row=row+1, col=ll+1,
                             )
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
                     name=f'{contrastive_label[0]}_{prompt_label[1]}',
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
                     name=f'{contrastive_label[0]}_{prompt_label[0]}',
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
                     name=f'{contrastive_label[1]}_{prompt_label[1]}',
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
                     name=f'{contrastive_label[1]}_{prompt_label[0]}',
                     marker_color='yellow',
                     ),
        row=row + 1, col=ll + 1,
    )

    fig.update_layout(height=330, width=1000)
    fig.show()
    fig.write_html('honest_lying_pca.html')

    return fig


def plot_one_layer_3d(activations_1, activations_2,
                      labels_1=None, labels_2=None,
                      contrastive_name=["honest", "lying"],
                      probability_1=None, probability_2=None,
                      color_by='label',
                      layer=16):

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
        label_text = np.append(label_text,
                               f'{contrastive_name[1]}_{labels_tf[ii + n_contrastive_data]}_{ii}_{probability_2[ii]}')

    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_1,
                                                                              activations_2), dim=0)

    pca = PCA(n_components=3)
    activations_pca_all = pca.fit_transform(activations_all[:, layer, :].cpu())
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=[f"layer {layer}"],
                        specs=[[{'type': 'scene'}]])
    df = {}
    df['label'] = labels_all
    df['pca0'] = activations_pca_all[:, 0]
    df['pca1'] = activations_pca_all[:, 1]
    df['pca2'] = activations_pca_all[:, 1]
    df['label_text'] = label_text
    fig.add_trace(
        go.Scatter3d(x=df['pca0'][:n_contrastive_data],
                     y=df['pca1'][:n_contrastive_data],
                     z=df['pca2'][:n_contrastive_data],
                     mode="markers",
                     showlegend=False,
                     marker=dict(
                         symbol="cross",
                         size=10,
                         line=dict(width=1, color="DarkSlateGrey"),
                         color=df['label'][:n_contrastive_data],
                         colorscale="Viridis",
                         colorbar=dict(
                             title="probability"
                         ),

                     ),
                     text=df['label_text'][:n_contrastive_data]),
        row=1, col=1,
    )

    # fig.add_trace(
    #     go.Scatter3d(x=df['pca0'][n_contrastive_data:n_contrastive_data * 2],
    #                  y=df['pca1'][n_contrastive_data:n_contrastive_data * 2],
    #                  z=df['pca2'][n_contrastive_data:n_contrastive_data * 2],
    #                  mode="markers",
    #                  showlegend=False,
    #                  marker=dict(
    #                      symbol="circle",
    #                      size=5,
    #                      line=dict(width=1, color="DarkSlateGrey"),
    #                      color=df['label'][n_contrastive_data:n_contrastive_data * 2],
    #                      colorscale="Viridis",
    #
    #                  ),
    #                  text=df['label_text'][n_contrastive_data:n_contrastive_data * 2]),
    #     row=1, col=1,
    # )

    fig.update_layout(height=500, width=500)
    fig.write_html(f'pca_layer_3d_layer_{layer}.html')
    return fig


def plot_one_layer_with_centroid_and_vector(activations_pca_all,
                                            centroid_honest_true, centroid_honest_false,
                                            centroid_lying_true, centroid_lying_false,
                                            centroid_vector_honest, centroid_vector_lying,
                                            labels,
                                            save_path,
                                            prompt_label=["honest", "lying"],
                                            layer=16):

    n_samples = len(labels)
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

    df = {}
    df['label'] = labels_all
    df['pca0'] = activations_pca_all[:, layer, 0]
    df['pca1'] = activations_pca_all[:, layer, 1]
    df['pca2'] = activations_pca_all[:, layer, 2]
    df['label_text'] = label_text

    # plot the centeroid vector
    # x y of quiver is the origin of the vector, u v is the end point of the vector
    fig = ff.create_quiver(x=[centroid_honest_false[layer, 0], centroid_lying_false[layer, 0]],
                           y=[centroid_honest_false[layer, 1], centroid_lying_false[layer, 1]],
                           u=[centroid_vector_honest[layer, 0], centroid_vector_lying[layer, 0]],
                           v=[centroid_vector_honest[layer, 1], centroid_vector_lying[layer, 1]],
                           line=dict(width=3, color='black'),
                           scale=1)
    fig.add_trace(
        go.Scatter(x=df['pca0'][:n_samples],
                   y=df['pca1'][:n_samples],
                   mode="markers",
                   showlegend=False,
                   marker=dict(
                         symbol="star",
                         size=8,
                         line=dict(width=1, color="DarkSlateGrey"),
                         color=df['label'][:n_samples],
                         opacity=0.5,
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
                         opacity=0.5,
                   ),
                   text=df['label_text'][n_samples:n_samples * 2]),
    )
    # plot centroid: centroid honest, true
    fig.add_trace(go.Scatter(
        x=[centroid_honest_true[layer, 0]],
        y=[centroid_honest_true[layer, 1]],
        marker=dict(
            symbol='star',
            size=10,
            color='orange'
        ),
        name='hoenst_true_centeroid'
    ))

    # plot centroid: centroid honest, false
    fig.add_trace(go.Scatter(
        x=[centroid_honest_false[layer, 0]],
        y=[centroid_honest_false[layer, 1]],
        marker=dict(
            symbol='star',
            size=10,
            color='blue'
        ),
        name='hoenst_false_centeroid'
    ))
    # plot centroid: centroid lying, true
    fig.add_trace(go.Scatter(
        x=[centroid_lying_true[layer, 0]],
        y=[centroid_lying_true[layer, 1]],
        marker=dict(
            symbol='circle',
            size=10,
            color='orange'
        ),
        name='lying_true_centeroid',
    ))
    # plot centroid: centroid lying, false
    fig.add_trace(go.Scatter(
        x=[centroid_lying_false[layer, 0]],
        y=[centroid_lying_false[layer, 1]],
        marker=dict(
            symbol='circle',
            size=10,
            color='blue'
        ),
        name='lying_false_centeroid',
    ))
    fig.update_layout(height=600, width=600,
                      title=dict(text=f"Layer {layer}", font=dict(size=30), automargin=True, yref='paper')
                      )
    fig.write_html(save_path + os.sep + f'pca_centroid_layer_{layer}.html')
    pio.write_image(fig, save_path + os.sep + f'pca_centroid_layer_{layer}.png',
                    scale=6)
    pio.write_image(fig, save_path + os.sep + f'pca_centroid_layer_{layer}.pdf',
                    scale=6)

    return fig
