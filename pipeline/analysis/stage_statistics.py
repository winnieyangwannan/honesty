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


# 0. Perform PCA layer by layer
def get_pca_layer_by_layer(activations_honest, activations_lying, n_layers, n_components=3):
    n_samples = activations_honest.shape[0]
    pca = PCA(n_components=n_components)
    activations_pca = np.zeros((n_samples*2, n_layers, n_components))
    for layer in range(n_layers):
        activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_honest,
                                                                                  activations_lying), dim=0)
        activations_pca[:, layer, :] = pca.fit_transform(activations_all[:, layer, :].cpu())
    return activations_pca


# 1. Stage 1: Separation between Honest and Lying
# Measurement: The distance between a pair of honest and lying prompt
# Future: Measure the within group (lying and honest) vs across group distance
def get_distance_pair_honest_lying(activations_all, activations_pca, n_layers, save_path):
    n_samples = int(activations_all.shape[0] / 2)
    dist_pair_pca = np.zeros((n_layers, n_samples))
    dist_pair_z_pca = np.zeros((n_layers, n_samples))
    dist_pair = np.zeros((n_layers, n_samples))
    dist_pair_z = np.zeros((n_layers, n_samples))
    activations_honest = activations_all[:n_samples, :, :]
    activations_lying = activations_all[n_samples:, :, :]

    for layer in range(n_layers):
        activations_pca_honest = activations_pca[:n_samples, layer, :]
        activations_pca_lying = activations_pca[n_samples:, layer, :]

        # original high dimensional space
        dist_between = cdist(activations_honest[:, layer, :].cpu(), activations_lying[:, layer, :].cpu()) # [n_samples by n_samples]
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
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Residual Stream Original', ' Residual Stream PCA',
                                        '', '')
                        )
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(dist_pair, axis=1),
                             mode='lines+markers',
                             showlegend=False,
                            ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(dist_pair_z, axis=1),
                             mode='lines+markers',
                             showlegend=False,
                             ), row=2, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(dist_pair_pca, axis=1),
                             mode='lines+markers',
                             showlegend=False,
                            ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(dist_pair_z_pca, axis=1),
                             mode='lines+markers',
                             showlegend=False,
                             ), row=2, col=2)
    fig.update_layout(height=1000, width=1200)
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
    pio.write_image(fig, save_path + os.sep + 'distance_pair.png',
                    scale=6)


# 2. Stage 2:  Separation between True and False
# Measurement: the distance between centroid between centroids of true and false
def get_dist_centroid_true_false(activations_all, activations_pca, labels, n_layers, save_path):
    n_samples = int(activations_all.shape[0] / 2)
    centroid_dist_honest = np.zeros((n_layers))
    centroid_dist_lying = np.zeros((n_layers))
    centroid_dist_honest_pca = np.zeros((n_layers))
    centroid_dist_lying_pca = np.zeros((n_layers))
    activations_honest = activations_all[:n_samples, :, :]
    activations_lying = activations_all[n_samples:, :, :]
    for layer in range(n_layers):
        activations_pca_honest = activations_pca[:n_samples, layer, :]
        activations_pca_lying = activations_pca[n_samples:, layer, :]

        centroid_dist_honest[layer] = get_centroid_dist(activations_honest[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]
        centroid_dist_lying[layer] = get_centroid_dist(activations_lying[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]

        centroid_dist_honest_pca[layer] = get_centroid_dist(activations_pca_honest[:, :], labels) # [n_samples by n_samples]
        centroid_dist_lying_pca[layer] = get_centroid_dist(activations_pca_lying[:, :], labels) # [n_samples by n_samples]

    # # plot
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Residual Stream Original', ' Residual Stream PCA'))
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=centroid_dist_honest,
                             name="honest",
                             mode='lines+markers',
                            ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=centroid_dist_lying,
                             name="lying",
                             mode='lines+markers',
                            ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=centroid_dist_honest_pca,
                             name="honest",
                             mode='lines+markers',
                           ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=centroid_dist_lying_pca,
                             name="lying",
                             mode='lines+markers',
                 ), row=1, col=2)
    fig['layout']['xaxis']['title'] = 'Layer'
    fig['layout']['xaxis2']['title'] = 'Layer'

    fig['layout']['yaxis']['title'] = 'Distance'
    fig['layout']['yaxis2']['title'] = ''
    fig.update_layout(height=400, width=800)
    fig.show()
    # fig.write_html(save_path + os.sep + 'centroid_distance_true_false.html')
    pio.write_image(fig, save_path
                    + os.sep + 'centroid_distance_true_false.png',
                    scale=6)


def get_centroid_dist(arr, labels):
    true_ind = [label == 1 for label in labels]
    false_ind = [label == 0 for label in labels]

    centroid_true = arr[true_ind, :].mean(axis=0)
    centroid_false = arr[false_ind, :].mean(axis=0)
    centroid_dist = math.dist(centroid_true, centroid_false)

    return centroid_dist


# 3. Stage 3: cosine similarity between the honest vector and lying vector
# Measurement:
def get_cos_sim_honest_lying_vector(activations_all, activations_pca, labels, n_layers, save_path):
    n_samples = int(activations_all.shape[0] / 2)
    n_components = activations_pca.shape[-1]
    angle_honest_lying = np.zeros((n_layers))
    angle_honest_lying_pca = np.zeros((n_layers))
    activations_honest = activations_all[:n_samples, :, :]
    activations_lying = activations_all[n_samples:, :, :]
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
        centroid_honest_true, centroid_honest_false, centroid_vector_honest = get_centroid_vector(activations_honest[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]
        centroid_lying_true, centroid_lying_false, centroid_vector_lying = get_centroid_vector(activations_lying[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]
        centroid_dir_honest = unit_vector(centroid_vector_honest)
        centroid_dir_lying = unit_vector(centroid_vector_lying)
        angle_honest_lying[layer] = cosine_similarity(centroid_dir_honest, centroid_dir_lying)
        # pca
        centroid_honest_true, centroid_honest_false, centroid_vector_honest = get_centroid_vector(activations_pca_honest, labels) # [n_samples by n_samples]
        centroid_lying_true, centroid_lying_false, centroid_vector_lying = get_centroid_vector(activations_pca_lying, labels) # [n_samples by n_samples]
        centroid_dir_honest = unit_vector(centroid_vector_honest)
        centroid_dir_lying = unit_vector(centroid_vector_lying)
        angle_honest_lying_pca[layer] = cosine_similarity(centroid_dir_honest, centroid_dir_lying)
        centroid_honest_true_pca_all[layer, :] = centroid_honest_true
        centroid_honest_false_pca_all[layer, :] = centroid_honest_false
        centroid_honest_vector_pca_all[layer, :] = centroid_vector_honest
        centroid_lying_true_pca_all[layer, :] = centroid_lying_true
        centroid_lying_false_pca_all[layer, :] = centroid_lying_false
        centroid_lying_vector_pca_all[layer, :] = centroid_vector_lying
    # # plot
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Residual Stream Original', ' Residual Stream PCA'))
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=angle_honest_lying,
                             mode='lines+markers',
                             showlegend=False,
                  ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=angle_honest_lying_pca,
                             mode='lines+markers',
                             showlegend=False,
                  ), row=1, col=2)
    fig.update_layout(height=400, width=1000)
    fig['layout']['xaxis']['title'] = 'Layer'
    fig['layout']['xaxis2']['title'] = 'Layer'

    fig['layout']['yaxis']['title'] = 'Cosine Similarity'
    fig['layout']['yaxis2']['title'] = ''

    fig['layout']['xaxis']['tickvals'] = np.arange(0, n_layers, 5)
    fig['layout']['xaxis2']['tickvals'] = np.arange(0, n_layers, 5)

    fig.show()
    # fig.write_html(save_path + os.sep + 'cos_sim_honest_lying.html')
    pio.write_image(fig, save_path + os.sep + 'cos_sim_honest_lying.png',
                    scale=6)
    return centroid_honest_true_pca_all, centroid_honest_false_pca_all, centroid_honest_vector_pca_all, \
           centroid_lying_true_pca_all, centroid_lying_false_pca_all, centroid_lying_vector_pca_all


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


def get_state_quantification(cfg, activations_honest, activations_lying, labels):
    """Run the full pipeline."""
    intervention = cfg.intervention
    save_path = os.path.join(cfg.artifact_path(), intervention, 'stage_stats')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    n_layers = activations_honest.shape[1]
    n_components = 3

    activations_all = torch.cat((activations_honest,
                                 activations_lying), dim=0)
    # 0. pca
    activations_pca = get_pca_layer_by_layer(activations_honest, activations_lying, n_layers, 
                                             n_components=n_components)

    # 1. Stage 1: Separation between Honest and Lying
    # Measurement: The distance between a pair of honest and lying prompt
    # Future: Measure the within group (lying and honest) vs across group distance
    get_distance_pair_honest_lying(activations_all, activations_pca, n_layers, save_path)

    # 2. Stage 2:  Separation between True and False
    # Measurement: the distance between centroid between centroids of true and false
    get_dist_centroid_true_false(activations_all, activations_pca, labels, n_layers, save_path)

    # 3. Stage 3: cosine similarity between the honest vector and lying vector
    # Measurement: cosine similarity between honest vector and lying vector 
    # honest vector is the centroid between honest true and honest false
    centroid_honest_true, centroid_honest_false, centroid_vector_honest, centroid_lying_true, centroid_lying_false, centroid_vector_lying = get_cos_sim_honest_lying_vector(activations_all, activations_pca, labels, n_layers, save_path)

    return activations_pca, centroid_honest_true, centroid_honest_false, centroid_vector_honest, centroid_lying_true, centroid_lying_false, centroid_vector_lying