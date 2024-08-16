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


# 0. Perform PCA layer by layer
def get_pca_layer_by_layer(activations_positive, activations_negative, n_layers,
                           n_components=3, save_plot=True):
    n_samples = activations_positive.shape[0]
    pca = PCA(n_components=n_components)
    activations_pca = np.zeros((n_samples*2, n_layers, n_components))
    for layer in range(n_layers):
        activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_positive,
                                                                                  activations_negative), dim=0)
        activations_pca[:, layer, :] = pca.fit_transform(activations_all[:, layer, :].cpu())
    return activations_pca


# 1. Stage 1: Separation between Honest and Lying
# Measurement: The distance between a pair of honest and lying prompt
# Future: Measure the within group (lying and honest) vs across group distance
def get_distance_pair_honest_lying(activations_all, activations_pca, n_layers, save_path,
                                   save_plot=True):
    n_samples = int(activations_all.shape[0] / 2)
    dist_pair_pca = np.zeros((n_layers, n_samples))
    dist_pair_z_pca = np.zeros((n_layers, n_samples))
    dist_pair = np.zeros((n_layers, n_samples))
    dist_pair_z = np.zeros((n_layers, n_samples))
    activations_positive = activations_all[:n_samples, :, :]
    activations_negative = activations_all[n_samples:, :, :]

    for layer in range(n_layers):
        activations_pca_honest = activations_pca[:n_samples, layer, :]
        activations_pca_lying = activations_pca[n_samples:, layer, :]

        # original high dimensional space
        dist_between = cdist(activations_positive[:, layer, :].cpu(), activations_negative[:, layer, :].cpu()) # [n_samples by n_samples]
        # zscore
        dist_z = stats.zscore(dist_between)
        # for the pair of the prompt with same statement, take the diagonal
        dist_pair[layer, :] = dist_between.diagonal()
        dist_pair_z[layer, :] = dist_z.diagonal()

        # pca
        dist_between_pca = cdist(activations_pca_honest[:, :], activations_pca_lying[:, :]) # [n_samples by n_samples]
        # zscore
        dist_z_pca = stats.zscore(dist_between_pca)
        # for the pair of the prompt with same statement, take the diagonal
        dist_pair_pca[layer, :] = dist_between_pca.diagonal()
        dist_pair_z_pca[layer, :] = dist_z_pca.diagonal()

    # # plot
    if save_plot:
        line_width = 2
        marker_size = 4
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Original High Dimensional Space', 'PCA',
                                            '', '')
                            )
        fig.add_trace(go.Scatter(
             x=np.arange(n_layers), y=np.mean(dist_pair, axis=1),
             mode='lines+markers',
             showlegend=False,
             marker=dict(size=marker_size),
             line=dict(color="royalblue", width=line_width)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
             x=np.arange(n_layers), y=np.mean(dist_pair_z, axis=1),
             mode='lines+markers',
             showlegend=False,
             marker=dict(size=marker_size),
             line=dict(color="royalblue", width=line_width)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
             x=np.arange(n_layers), y=np.mean(dist_pair_pca, axis=1),
             mode='lines+markers',
             showlegend=False,
             marker=dict(size=marker_size),
             line=dict(color="royalblue", width=line_width)
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
             x=np.arange(n_layers), y=np.mean(dist_pair_z_pca, axis=1),
             mode='lines+markers',
             showlegend=False,
             marker=dict(size=marker_size),
             line=dict(color="royalblue", width=line_width)
        ), row=2, col=2)
        fig.update_xaxes(tickvals=np.arange(0, n_layers, 5))
        fig.update_layout(height=500, width=500)
        fig['layout']['xaxis']['title'] = 'Layer'
        fig['layout']['xaxis2']['title'] = 'Layer'
        fig['layout']['xaxis3']['title'] = 'Layer'
        fig['layout']['xaxis4']['title'] = 'Layer'

        fig['layout']['yaxis']['title'] = 'Distance'
        fig['layout']['yaxis2']['title'] = ''
        fig['layout']['yaxis3']['title'] = 'Distance (z-scored)'
        fig['layout']['yaxis4']['title'] = ''

        fig.show()
        # fig.write_html(save_path + os.sep + 'distance_pair.html')
        pio.write_image(fig, save_path + os.sep + 'stage_1_distance_pair.png',
                        scale=6)

    stage_1 = {
        'dist_pair': dist_pair,
        'dist_pair_pca': dist_pair_pca,
        'dist_pair_z': dist_pair_z,
        'dist_pair_z_pca': dist_pair_z_pca
    }
    # return dist_pair, dist_pair_pca, dist_pair_z, dist_pair_z_pca
    return stage_1


# 2. Stage 2:  Separation between True and False
# Measurement: the distance between centroid between centroids of true and false
def get_dist_centroid_true_false(activations_all, activations_pca, labels, n_layers, save_path, save_plot=True):
    n_samples = int(activations_all.shape[0] / 2)
    centroid_dist_honest = np.zeros((n_layers))
    centroid_dist_lying = np.zeros((n_layers))
    centroid_dist_honest_pca = np.zeros((n_layers))
    centroid_dist_lying_pca = np.zeros((n_layers))
    centroid_dist_honest_z = np.zeros((n_layers))
    centroid_dist_lying_z = np.zeros((n_layers))
    centroid_dist_honest_pca_z = np.zeros((n_layers))
    centroid_dist_lying_pca_z = np.zeros((n_layers))
    activations_positive = activations_all[:n_samples, :, :]

    activations_negative = activations_all[n_samples:, :, :]
    for layer in range(n_layers):
        activations_pca_honest = activations_pca[:n_samples, layer, :]
        activations_pca_lying = activations_pca[n_samples:, layer, :]

        centroid_dist_honest[layer] = get_centroid_dist(activations_positive[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]
        centroid_dist_lying[layer] = get_centroid_dist(activations_negative[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]

        centroid_dist_honest_pca[layer] = get_centroid_dist(activations_pca_honest[:, :], labels) # [n_samples by n_samples]
        centroid_dist_lying_pca[layer] = get_centroid_dist(activations_pca_lying[:, :], labels) # [n_samples by n_samples]
        #

    # # plot
    if save_plot:
        line_width = 3
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Original High Dimensional Space', 'PCA'))
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=centroid_dist_honest,
                                 name="honest",
                                 mode='lines+markers',
                                 line=dict(color="royalblue", width=line_width)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=centroid_dist_lying,
                                 name="lying",
                                 mode='lines+markers',
                                 line=dict(color="firebrick", width=line_width)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=centroid_dist_honest_pca,
                                 showlegend=False,
                                 mode='lines+markers',
                                 line=dict(color="royalblue", width=line_width)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=centroid_dist_lying_pca,
                                 showlegend=False,
                                 mode='lines+markers',
                                 line=dict(color="firebrick", width=line_width)
        ), row=2, col=1)
        fig['layout']['xaxis2']['title'] = 'Layer'
        fig['layout']['yaxis']['title'] = 'Distance'
        fig['layout']['yaxis2']['title'] = 'Distance'

        fig.update_xaxes(tickvals=np.arange(0, n_layers, 5))
        fig.update_layout(height=500, width=700)
        fig.show()
        # fig.write_html(save_path + os.sep + 'centroid_distance_true_false.html')
        pio.write_image(fig, save_path
                        + os.sep + 'state_2_centroid_distance_true_false.png',
                        scale=6)
    stage_2 = {
               'centroid_dist_honest': centroid_dist_honest,
               'centroid_dist_lying': centroid_dist_lying,
               'centroid_dist_honest_pca': centroid_dist_honest_pca,
               'centroid_dist_lying_pca': centroid_dist_lying_pca

    }
    return stage_2


def get_centroid_dist(arr, labels):
    true_ind = [label == 1 for label in labels]
    false_ind = [label == 0 for label in labels]

    centroid_true = arr[true_ind, :].mean(axis=0)
    centroid_false = arr[false_ind, :].mean(axis=0)
    centroid_dist = math.dist(centroid_true, centroid_false)

    return centroid_dist


# 3. Stage 3: cosine similarity between the honest vector and lying vector
# Measurement:
def get_cos_sim_honest_lying_vector(activations_all, activations_pca, labels, n_layers, save_path,
                                    save_plot=True):
    n_samples = int(activations_all.shape[0] / 2)
    n_components = activations_pca.shape[-1]
    cos_honest_lying = np.zeros((n_layers))
    cos_honest_lying_pca = np.zeros((n_layers))
    activations_positive = activations_all[:n_samples, :, :]
    activations_negative = activations_all[n_samples:, :, :]
    centroid_lying_true_pca_all = np.zeros((n_layers, n_components))
    centroid_lying_false_pca_all = np.zeros((n_layers, n_components))
    centroid_lying_vector_pca_all = np.zeros((n_layers, n_components))
    centroid_honest_true_pca_all = np.zeros((n_layers, n_components))
    centroid_honest_false_pca_all = np.zeros((n_layers, n_components))
    centroid_honest_vector_pca_all = np.zeros((n_layers, n_components))

    for layer in range(n_layers):
        activations_pca_honest = activations_pca[:n_samples, layer, :]
        activations_pca_lying = activations_pca[n_samples:, layer, :]
        # original high d
        centroid_honest_true, centroid_honest_false, centroid_vector_honest = get_centroid_vector(activations_positive[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]
        centroid_lying_true, centroid_lying_false, centroid_vector_lying = get_centroid_vector(activations_negative[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]
        centroid_dir_honest = unit_vector(centroid_vector_honest)
        centroid_dir_lying = unit_vector(centroid_vector_lying)
        cos_honest_lying[layer] = cosine_similarity(centroid_dir_honest, centroid_dir_lying)
        # pca
        centroid_honest_true, centroid_honest_false, centroid_vector_honest = get_centroid_vector(activations_pca_honest, labels) # [n_samples by n_samples]
        centroid_lying_true, centroid_lying_false, centroid_vector_lying = get_centroid_vector(activations_pca_lying, labels) # [n_samples by n_samples]
        centroid_dir_honest = unit_vector(centroid_vector_honest)
        centroid_dir_lying = unit_vector(centroid_vector_lying)
        cos_honest_lying_pca[layer] = cosine_similarity(centroid_dir_honest, centroid_dir_lying)
        centroid_honest_true_pca_all[layer, :] = centroid_honest_true
        centroid_honest_false_pca_all[layer, :] = centroid_honest_false
        centroid_honest_vector_pca_all[layer, :] = centroid_vector_honest
        centroid_lying_true_pca_all[layer, :] = centroid_lying_true
        centroid_lying_false_pca_all[layer, :] = centroid_lying_false
        centroid_lying_vector_pca_all[layer, :] = centroid_vector_lying
    # # plot
    if save_plot:

        line_width = 3
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Original High Dimensional Space', 'PCA'))
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=cos_honest_lying,
                                 mode='lines+markers',
                                 showlegend=False,
                                 line=dict(color="royalblue", width=line_width),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=cos_honest_lying_pca,
                                 mode='lines+markers',
                                 showlegend=False,
                                 line=dict(color="royalblue", width=line_width),
        ), row=2, col=1)


        fig.update_layout(height=700, width=500)
        fig['layout']['xaxis']['title'] = 'Layer'
        fig['layout']['xaxis2']['title'] = 'Layer'

        fig['layout']['yaxis']['title'] = 'Cosine Similarity'
        fig['layout']['yaxis2']['title'] = 'Cosine Similarity'

        fig['layout']['xaxis']['tickvals'] = np.arange(0, n_layers, 5)
        fig['layout']['xaxis2']['tickvals'] = np.arange(0, n_layers, 5)

        fig['layout']['yaxis']['tickvals'] = np.arange(-1, 1.2, 0.5)
        fig['layout']['yaxis2']['tickvals'] = np.arange(-1, 1.2, 0.5)

        fig['layout']['yaxis']['range'] = [-1, 1.2]
        fig['layout']['yaxis2']['range'] = [-1,1.2]

        fig.show()
        # fig.write_html(save_path + os.sep + 'cos_sim_honest_lying.html')
        pio.write_image(fig, save_path + os.sep + 'stage_3_cos_sim_honest_lying.png',
                        scale=6)
    stage_3 = {
               'centroid_honest_true_pca_all': centroid_honest_true_pca_all,
               'centroid_honest_false_pca_all': centroid_honest_false_pca_all,
               'centroid_honest_vector_pca_all': centroid_honest_vector_pca_all,
               'centroid_lying_true_pca_all': centroid_lying_true_pca_all,
               'centroid_lying_false_pca_all': centroid_lying_false_pca_all,
               'centroid_lying_vector_pca_all': centroid_lying_vector_pca_all,
               'cos_honest_lying': cos_honest_lying,
               'cos_honest_lying_pca': cos_honest_lying_pca
    }
    return stage_3


def get_centroid_vector(arr, labels):
    true_ind = [label == 1 for label in labels]
    false_ind = [label == 0 for label in labels]

    centroid_true = arr[true_ind, :].mean(axis=0)
    centroid_false = arr[false_ind, :].mean(axis=0)
    centroid_vector = centroid_true - centroid_false

    return centroid_true, centroid_false, centroid_vector


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def cosine_similarity(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)


def get_state_quantification(cfg, activations_positive, activations_negative, labels,
                             save_plot=True):
    """Run the full pipeline."""
    intervention = cfg.intervention

    if intervention == "no_intervention":
        save_path = os.path.join(cfg.artifact_path(), 'stage_stats')
    else:
        save_path = os.path.join(cfg.artifact_path(), intervention, 'stage_stats')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    n_layers = activations_positive.shape[1]
    n_components = 3

    activations_all = torch.cat((activations_positive,
                                 activations_negative), dim=0)
    # 0. pca
    activations_pca = get_pca_layer_by_layer(activations_positive, activations_negative, n_layers, 
                                             n_components=n_components)

    # 1. Stage 1: Separation between Honest and Lying
    # Measurement: The distance between a pair of honest and lying prompt
    # Future: Measure the within group (lying and honest) vs across group distance
    stage_1 = get_distance_pair_honest_lying(activations_all, activations_pca, n_layers, save_path,
                                             save_plot=save_plot)

    # 2. Stage 2:  Separation between True and False
    # Measurement: the distance between centroid between centroids of true and false
    stage_2 = get_dist_centroid_true_false(activations_all, activations_pca, labels, n_layers, save_path,
                                           save_plot=save_plot)

    # 3. Stage 3: cosine similarity between the honest vector and lying vector
    # Measurement: cosine similarity between honest vector and lying vector 
    # honest vector is the centroid between honest true and honest false
    stage_3 = get_cos_sim_honest_lying_vector(activations_all, activations_pca, labels, n_layers, save_path,
                                              save_plot=save_plot)

    stage_stats = {
      'stage_1': stage_1,
      'stage_2': stage_2,
      'stage_3': stage_3
    }

    return stage_stats


###################
def plot_stage_3_stats_original_intervention(cfg, stage_3_original, stage_3_intervention, n_layers, save_path):

    source_layer = cfg.source_layer
    target_layer_s = cfg.target_layer_s
    target_layer_e = cfg.target_layer_e

    # plot stage 3
    line_width =3
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Original high dimensional space', 'PCA',
                                        '', '')
                        )

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_original['cos_honest_lying'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_intervention['cos_honest_lying'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width, dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_original['cos_honest_lying_pca'],
                             mode='lines+markers',
                             name="Original",
                             line=dict(color="royalblue", width=line_width)

    ), row=2, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_intervention['cos_honest_lying_pca'],
                             mode='lines+markers',
                             name="Intervention",
                             line=dict(color="royalblue", width=line_width, dash='dot')

    ), row=2, col=1)

    fig.update_xaxes(tickvals=np.arange(0, n_layers, 5))

    fig.update_layout(height=800, width=1000)
    fig['layout']['xaxis']['title'] = 'Layer'
    fig['layout']['xaxis2']['title'] = 'Layer'
    fig['layout']['yaxis']['title'] = 'Cosine similarity'
    fig['layout']['yaxis2']['title'] = 'Cosine similarity'

    fig['layout']['yaxis']['tickvals'] = np.arange(-1, 1.2, 0.5)
    fig['layout']['yaxis2']['tickvals'] = np.arange(-1, 1.2, 0.5)
    fig['layout']['yaxis']['range'] = [-1, 1.2]
    fig['layout']['yaxis2']['range'] = [-1, 1.2]
    fig.show()
    # fig.write_html(save_path + os.sep + 'distance_pair.html')
    pio.write_image(fig, save_path + os.sep + 'stage_3_cosine_similarity_layer_' + str(source_layer) + '_' + str(target_layer_s) + '_' + str(target_layer_e) + '.png',
                    scale=6)
    

def plot_stage_2_stats_original_intervention(cfg, stage_2_original, stage_2_intervention, n_layers, save_path):
    # plot stage 2
    line_width =3
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Original high dimensional space', 'PCA',
                                        '', '')
                        )

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_original['centroid_dist_honest'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_original['centroid_dist_lying'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="firebrick", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_intervention['centroid_dist_honest'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width, dash='dot')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_intervention['centroid_dist_lying'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="firebrick", width=line_width, dash='dot')
    ), row=1, col=1)
 
    # PCA
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_original['centroid_dist_honest_pca'],
                             mode='lines+markers',
                             name="Original_honest",
                             line=dict(color="royalblue", width=line_width)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_original['centroid_dist_lying_pca'],
                             mode='lines+markers',
                             name="Original_lying",
                             line=dict(color="firebrick", width=line_width)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_intervention['centroid_dist_honest_pca'],
                             mode='lines+markers',
                             name="Intervention_honest",
                             line=dict(color="royalblue", width=line_width, dash='dot')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_intervention['centroid_dist_lying_pca'],
                             mode='lines+markers',
                             name="Intervention_lying",
                             line=dict(color="firebrick", width=line_width, dash='dot')
    ), row=2, col=1)
  
    fig.update_xaxes(tickvals=np.arange(0, n_layers, 5))
    fig.update_layout(height=800, width=1000)
    fig['layout']['xaxis']['title'] = 'Layer'
    fig['layout']['xaxis2']['title'] = 'Layer'
    fig['layout']['yaxis']['title'] = 'Distance'
    fig['layout']['yaxis2']['title'] = 'Distance'

    fig.show()

    source_layer = cfg.source_layer
    target_layer_s = cfg.target_layer_s
    target_layer_e = cfg.target_layer_e
    # fig.write_html(save_path + os.sep + 'distance_pair.html')
    pio.write_image(fig, save_path + os.sep + 'stage_2_centroid_distance_true_false_layer_' +
                    str(source_layer) + '_' + str(target_layer_s) + '_' + str(target_layer_e) +'.png',
                    scale=6)
    
    
def plot_stage_1_stats_original_intervention(cfg, stage_1_original, stage_1_intervention, n_layers, save_path):
    # plot stage 1
    line_width =3
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Original high dimensional space', 'PCA',
                                        '', '')
                        )

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(stage_1_original['dist_pair_z'], axis=1),
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(stage_1_intervention['dist_pair_z'], axis=1),
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width, dash="dot")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(stage_1_original['dist_pair_z_pca'], axis=1),
                             mode='lines+markers',
                             name="Original",
                             line=dict(color="royalblue", width=line_width)

    ), row=2, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(stage_1_intervention['dist_pair_z_pca'], axis=1),
                             mode='lines+markers',
                             name="Intervention",
                             line=dict(color="royalblue", width=line_width, dash="dot")

    ), row=2, col=1)
    fig.update_xaxes(tickvals=np.arange(0, n_layers, 5))

    fig.update_layout(height=800, width=1000)
    fig['layout']['xaxis']['title'] = 'Layer'
    fig['layout']['xaxis2']['title'] = 'Layer'
    fig['layout']['yaxis']['title'] = 'Distance (z_scored)'
    fig['layout']['yaxis2']['title'] = ''

    fig.show()

    source_layer = cfg.source_layer
    target_layer_s = cfg.target_layer_s
    target_layer_e = cfg.target_layer_e

    # fig.write_html(save_path + os.sep + 'distance_pair.html')
    pio.write_image(fig, save_path + os.sep + 'stage_1_distance_pair_layer_' +
                    str(source_layer) + '_' + str(target_layer_s) + '_' + str(target_layer_e) + '.png',
                    scale=6)


def plot_stage_quantification_original_intervention(cfg, stage_stats_original, stage_stats_intervention,
                                                    n_layers, save_path):

    source_layer = cfg.source_layer
    # 1. Stage 1
    stage_1_original = stage_stats_original['stage_1']
    stage_1_intervention = stage_stats_intervention['stage_1']

    # 2. Stage 2
    stage_2_original = stage_stats_original['stage_2']
    stage_2_intervention = stage_stats_intervention['stage_2']

    # 3. Stage 3
    stage_3_original = stage_stats_original['stage_3']
    stage_3_intervention = stage_stats_intervention['stage_3']

    plot_stage_1_stats_original_intervention(cfg, stage_1_original, stage_1_intervention, n_layers, save_path)
    plot_stage_2_stats_original_intervention(cfg, stage_2_original, stage_2_intervention, n_layers, save_path)
    plot_stage_3_stats_original_intervention(cfg, stage_3_original, stage_3_intervention, n_layers, save_path)

