import os
import argparse
from pipeline.model_utils.model_factory import construct_model_base
import pickle
import csv
import math
from tqdm import tqdm
from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
from sklearn.metrics.pairwise import pairwise_distances

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_quiver
import plotly.figure_factory as ff
import plotly.io as pio


from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import PCA
import numpy as np
import torch

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
from scipy.spatial.distance import cdist
from scipy import stats


# 0. Perform PCA layer by layer

def get_pca_layer_by_layer_dunn(activations_all,
                                n_components=3):
    n_layers = activations_all.shape[2]
    n_samples = activations_all.shape[1]
    n_contrastive_group = activations_all.shape[0]
    pca = PCA(n_components=n_components)

    activations_pca = np.zeros((n_contrastive_group, n_samples, n_layers, n_components))
    for group in range(n_contrastive_group):
        for layer in range(n_layers):
            activations_pca[group, :, layer, :] = pca.fit_transform(activations_all[group, :, layer, :].cpu().numpy())

    return activations_pca


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


# 1. Stage 1: Separation between positive and negative (or harmful and harmless)
# Measurement: The distance between a pair of positive and negative prompt
# Future: Measure the within group (negative and positive) vs across group distance
def get_distance_pair_contrastive(activations_all, activations_pca, n_layers, save_path,
                                   contrastive_label=['HHH', 'evil_confidant'], save_plot=True):
    n_samples = int(activations_all.shape[0] / 2)
    dist_pair_pca = np.zeros((n_layers, n_samples))
    dist_pair_z_pca = np.zeros((n_layers, n_samples))
    dist_pair = np.zeros((n_layers, n_samples))
    dist_pair_z = np.zeros((n_layers, n_samples))
    activations_positive = activations_all[:n_samples, :, :]
    activations_negative = activations_all[n_samples:, :, :]

    for layer in range(n_layers):
        activations_pca_positive = activations_pca[:n_samples, layer, :]
        activations_pca_negative = activations_pca[n_samples:, layer, :]

        # original high dimensional space
        dist_between = cdist(activations_positive[:, layer, :].cpu(), activations_negative[:, layer, :].cpu()) # [n_samples by n_samples]
        # zscore
        dist_z = stats.zscore(dist_between)
        # for the pair of the prompt with same statement, take the diagonal
        dist_pair[layer, :] = dist_between.diagonal()
        dist_pair_z[layer, :] = dist_z.diagonal()

        # pca
        dist_between_pca = cdist(activations_pca_positive[:, :], activations_pca_negative[:, :]) # [n_samples by n_samples]
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
        pio.write_image(fig, save_path + os.sep + 'stage_1_distance_pair_' +
                        f'_{contrastive_label[0]}_{contrastive_label[1]}' + '.png',
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
def get_intra_cluster_distance(cluster_1, cluster_2):

    """
    intra_cluster_distance measures the compactness or cohesion of data points within a cluster.
    The smaller the intra-cluster distance, the more similar and tightly packed the data points are within the cluster.

    input:
        cluster_1 [n_data, d_data]
        cluster_2 [n_data, d_data]
    output:
        cluster_1 [n_data, n_data]
        cluster_2 [n_data, n_data]
    """

    distance_intra_1 = pairwise_distances(cluster_1, metric='euclidean')
    distance_intra_2 = pairwise_distances(cluster_2, metric='euclidean')
    # distance_intra_1 = np.tril(distance_intra_1, k=-1)
    # distance_intra_2 = np.tril(distance_intra_2, k=-1)

    return distance_intra_1, distance_intra_2


def intra_cluster_distance_high_pca(activations_all, activations_pca):
    """
    measures the separation or dissimilarity between clusters.
    The larger the inter-cluster distance, the more distinct and well-separated the clusters are from each other.
    """
    activations_all_1 = activations_all[0]
    activations_all_2 = activations_all[1]

    activations_pca_1 = activations_pca[0]
    activations_pca_2 = activations_pca[1]

    distance_intra_cluster_1, distance_intra_cluster_2 = get_intra_cluster_distance(activations_all_1,
                                                                                    activations_all_2)
    distance_intra_cluster_1_pca, distance_intra_cluster_2_pca = get_intra_cluster_distance(activations_pca_1,
                                                                                            activations_pca_2)
    distance_intra = np.stack((distance_intra_cluster_1, distance_intra_cluster_2), axis=0)
    distance_intra_pca = np.stack((distance_intra_cluster_1_pca, distance_intra_cluster_2_pca), axis=0)

    return distance_intra, distance_intra_pca


def get_inter_cluster_distance(cluster_1, cluster_2):
    n_data = cluster_1.shape[0]
    cluster_all = np.concatenate((cluster_1, cluster_2), axis=0)
    distance_all = pairwise_distances(cluster_all, metric='euclidean')
    distance_inter_cluster = distance_all[n_data:, :n_data]

    return distance_inter_cluster


def inter_cluster_distance_high_pca(activations_all, activations_pca):
    activations_all_1 = activations_all[0]
    activations_all_2 = activations_all[1]

    activations_pca_1 = activations_pca[0]
    activations_pca_2 = activations_pca[1]

    distance_inter_cluster = get_inter_cluster_distance(activations_all_1, activations_all_2)
    distance_inter_cluster_pca = get_inter_cluster_distance(activations_pca_1, activations_pca_2)

    return distance_inter_cluster, distance_inter_cluster_pca


def get_dunn_index(distance_intra_cluster, distance_inter_cluster):
    """
    Index = min_intercluster_distance / max_intracluster_distance
    see nice explanation here: https://medium.com/@Suraj_Yadav/understanding-intra-cluster-distance-inter-cluster-distance-and-dun-index-a-comprehensive-guide-a8de726f5769 )

    min_intercluster_distance: The minimum distance between any pair of data points from different clusters.
    max_intracluster_distance: The maximum distance between any pair of data points within the same cluster.
    the Dunn Index compares the smallest distance between two clusters with the largest distance within a cluster. A higher Dunn Index value indicates a better clustering solution with more distinct and well-separated clusters.

    """
    # distance_intra_cluster_norm = MinMaxScaler().fit_transform(distance_intra_cluster.reshape(-1, 1))
    # distance_inter_cluster_norm = MinMaxScaler().fit_transform(distance_inter_cluster.reshape(-1, 1))

    # distance_intra_cluster_norm = stats.zscore(distance_intra_cluster, axis=None)
    # distance_inter_cluster_norm = stats.zscore(distance_inter_cluster, axis=None)

    max_intracluster_distance = np.mean(distance_intra_cluster)
    min_intercluster_distance = np.mean(distance_inter_cluster)

    # max_intracluster_distance = np.max(distance_intra_cluster)
    # min_intercluster_distance = np.min(distance_inter_cluster)
    index = min_intercluster_distance/max_intracluster_distance
    return index


def dunn_index_contrastive(distance_intra_cluster, distance_intra_cluster_pca,
                           distance_inter_cluster, distance_inter_cluster_pca):

    dunn_index = get_dunn_index(distance_intra_cluster, distance_inter_cluster)
    dunn_index_pca = get_dunn_index(distance_intra_cluster_pca, distance_inter_cluster_pca)

    return dunn_index, dunn_index_pca


def plot_stage_2_dunn(dunn_index_all, dunn_index_pca, contrastive_type):
    n_layers = dunn_index_all.shape[1]
    line_width = 2
    marker_size = 4
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Original High Dimensional Space', 'PCA',
                                       )
                        )
    fig.add_trace(go.Scatter(
        x=np.arange(n_layers), y=np.mean(dunn_index_all[0], axis=1),
        mode='lines+markers',
        name=contrastive_type[0],
        showlegend=False,
        marker=dict(size=marker_size),
        line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.arange(n_layers), y=np.mean(dunn_index_all[1], axis=1),
        mode='lines+markers',
        name=contrastive_type[1],
        showlegend=False,
        marker=dict(size=marker_size),
        line=dict(color="firebrick", width=line_width)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=np.arange(n_layers), y=np.mean(dunn_index_pca[0], axis=1),
        mode='lines+markers',
        name=contrastive_type[0],
        showlegend=False,
        marker=dict(size=marker_size),
        line=dict(color="royalblue", width=line_width)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=np.arange(n_layers), y=np.mean(dunn_index_pca[1], axis=1),
        mode='lines+markers',
        name=contrastive_type[1],
        showlegend=False,
        marker=dict(size=marker_size),
        line=dict(color="firebrick", width=line_width)
    ), row=1, col=2)
    fig.show()

    return fig


def stage_2_get_distance_contrastive_dunn_index(activations_all, activations_all_pca,
                                                labels_all,
                                                contrastive_type, save_path
                                                 ):
    activations_all = activations_all.cpu().numpy()
    n_data = int(activations_all[0].shape[0]/2)
    n_layers = activations_all.shape[2]

    dunn_index_all = np.zeros((2, n_layers, 1))
    dunn_index_pca_all = np.zeros((2, n_layers, 1))
    for ii in range(2):
        # first half of data
        activations_positive = activations_all[ii][:n_data, :, :]
        activations_positive_pca = activations_all_pca[ii][:n_data, :, :]
        labels_positive = labels_all[ii][:n_data]
        # second half of data
        activations_negative = activations_all[ii][n_data:, :, :]
        activations_negative_pca = activations_all_pca[ii][n_data:, :, :]
        labels_negative = labels_all[ii][n_data:]
        # concatenate two contrastive group
        activations = np.stack((activations_positive, activations_negative))
        activations_pca = np.stack((activations_positive_pca, activations_negative_pca))
        labels = np.stack((labels_positive, labels_negative))


        dist_all = np.zeros((n_layers, 2*n_data, 2*n_data))
        dist_all_pca = np.zeros((n_layers, 2*n_data, 2*n_data))
        for layer in range(n_layers):
            # dist = pairwise_distances(activations[:, layer, :])
            # dist_pca = pairwise_distances(activations_pca[:, layer, :])
            # dist_all[layer] = dist
            # dist_all_pca[layer] = dist_pca

            # step 1: get intra-cluster distance (within one contrastive cluster)
            distance_intra_cluster, distance_intra_cluster_pca = intra_cluster_distance_high_pca(activations[:, :, layer, :],
                                                                                                 activations_pca[:, :, layer, :])

            # step 2: get inter-cluster distance (between contrastive clusters)
            distance_inter_cluster, distance_inter_cluster_pca = inter_cluster_distance_high_pca(activations_all[:, :, layer, :],
                                                                                                 activations_pca[:, :, layer, :])

            # step 3: calculate Dunn-index (see a nice explanation here: https://medium.com/@Suraj_Yadav/understanding-intra-cluster-distance-inter-cluster-distance-and-dun-index-a-comprehensive-guide-a8de726f5769 )
            dunn_index, dunn_index_pca = dunn_index_contrastive(distance_intra_cluster, distance_intra_cluster_pca,
                                                                distance_inter_cluster, distance_inter_cluster_pca)
            dunn_index_all[ii, layer] = dunn_index
            dunn_index_pca_all[ii, layer] = dunn_index_pca

    fig = plot_stage_2_dunn(dunn_index_all, dunn_index_pca_all, contrastive_type=contrastive_type)
    fig.write_html(save_path + os.sep + 'stage_2_dunn_index_' +
                   f'_{contrastive_type[0]}_{contrastive_type[1]}' + '.html')
    pio.write_image(fig, save_path + os.sep + 'stage_2_dunn_index_' +
                    f'_{contrastive_type[0]}_{contrastive_type[1]}' + '.png',
                    scale=6)

    stage_2_dunn = {
        'dunn_index_all': dunn_index_all,
        'dunn_index_pca_all': dunn_index_pca_all,
    }

    return stage_2_dunn


def get_dist_centroid_true_false(activations_all, activations_pca, labels,
                                 n_layers, save_path,
                                 contrastive_label=['HHH', 'evil_confidant'],
                                 save_plot=True):
    n_samples = int(activations_all.shape[0] / 2)
    centroid_dist_positive = np.zeros((n_layers))
    centroid_dist_negative = np.zeros((n_layers))
    centroid_dist_positive_pca = np.zeros((n_layers))
    centroid_dist_negative_pca = np.zeros((n_layers))
    centroid_dist_positive_z = np.zeros((n_layers))
    centroid_dist_negative_z = np.zeros((n_layers))
    centroid_dist_positive_pca_z = np.zeros((n_layers))
    centroid_dist_negative_pca_z = np.zeros((n_layers))
    activations_positive = activations_all[:n_samples, :, :]

    activations_negative = activations_all[n_samples:, :, :]
    for layer in range(n_layers):
        activations_pca_positive = activations_pca[:n_samples, layer, :]
        activations_pca_negative = activations_pca[n_samples:, layer, :]

        centroid_dist_positive[layer] = get_centroid_dist(activations_positive[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]
        centroid_dist_negative[layer] = get_centroid_dist(activations_negative[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]

        centroid_dist_positive_pca[layer] = get_centroid_dist(activations_pca_positive[:, :], labels) # [n_samples by n_samples]
        centroid_dist_negative_pca[layer] = get_centroid_dist(activations_pca_negative[:, :], labels) # [n_samples by n_samples]
        #

    # # plot
    if save_plot:
        line_width = 3
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Original High Dimensional Space', 'PCA'))
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=centroid_dist_positive,
                                 name=contrastive_label[0],
                                 mode='lines+markers',
                                 line=dict(color="royalblue", width=line_width)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=centroid_dist_negative,
                                 name=contrastive_label[1],
                                 mode='lines+markers',
                                 line=dict(color="firebrick", width=line_width)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=centroid_dist_positive_pca,
                                 showlegend=False,
                                 mode='lines+markers',
                                 line=dict(color="royalblue", width=line_width)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=centroid_dist_negative_pca,
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
        pio.write_image(fig, save_path + os.sep + 'state_2_centroid_distance_true_false_' +
                        f'_{contrastive_label[0]}_{contrastive_label[1]}' + '.png',
                        scale=6)
    stage_2 = {
               'centroid_dist_positive': centroid_dist_positive,
               'centroid_dist_negative': centroid_dist_negative,
               'centroid_dist_positive_pca': centroid_dist_positive_pca,
               'centroid_dist_negative_pca': centroid_dist_negative_pca

    }
    return stage_2


def get_centroid_dist(arr, labels):
    true_ind = [label == 1 for label in labels]
    false_ind = [label == 0 for label in labels]

    centroid_true = arr[true_ind, :].mean(axis=0)
    centroid_false = arr[false_ind, :].mean(axis=0)
    centroid_dist = math.dist(centroid_true, centroid_false)

    return centroid_dist


# 3. Stage 3: cosine similarity between the positive vector and negative vector
# Measurement:
def get_cos_sim_positive_negative_vector(activations_all, activations_pca, labels, n_layers, save_path,
                                    contrastive_label=['HHH', 'evil_confidant'], save_plot=True):
    n_samples = int(activations_all.shape[0] / 2)
    n_components = activations_pca.shape[-1]
    cos_positive_negative = np.zeros((n_layers))
    cos_positive_negative_pca = np.zeros((n_layers))
    activations_positive = activations_all[:n_samples, :, :]
    activations_negative = activations_all[n_samples:, :, :]
    centroid_negative_true_pca_all = np.zeros((n_layers, n_components))
    centroid_negative_false_pca_all = np.zeros((n_layers, n_components))
    centroid_negative_vector_pca_all = np.zeros((n_layers, n_components))
    centroid_positive_true_pca_all = np.zeros((n_layers, n_components))
    centroid_positive_false_pca_all = np.zeros((n_layers, n_components))
    centroid_positive_vector_pca_all = np.zeros((n_layers, n_components))

    for layer in range(n_layers):
        activations_pca_positive = activations_pca[:n_samples, layer, :]
        activations_pca_negative = activations_pca[n_samples:, layer, :]
        # original high d
        centroid_positive_true, centroid_positive_false, centroid_vector_positive = get_centroid_vector(activations_positive[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]
        centroid_negative_true, centroid_negative_false, centroid_vector_negative = get_centroid_vector(activations_negative[:, layer, :].cpu().numpy(), labels) # [n_samples by n_samples]
        centroid_dir_positive = unit_vector(centroid_vector_positive)
        centroid_dir_negative = unit_vector(centroid_vector_negative)
        cos_positive_negative[layer] = cosine_similarity(centroid_dir_positive, centroid_dir_negative)
        # pca
        centroid_positive_true, centroid_positive_false, centroid_vector_positive = get_centroid_vector(activations_pca_positive, labels) # [n_samples by n_samples]
        centroid_negative_true, centroid_negative_false, centroid_vector_negative = get_centroid_vector(activations_pca_negative, labels) # [n_samples by n_samples]
        centroid_dir_positive = unit_vector(centroid_vector_positive)
        centroid_dir_negative = unit_vector(centroid_vector_negative)
        cos_positive_negative_pca[layer] = cosine_similarity(centroid_dir_positive, centroid_dir_negative)
        centroid_positive_true_pca_all[layer, :] = centroid_positive_true
        centroid_positive_false_pca_all[layer, :] = centroid_positive_false
        centroid_positive_vector_pca_all[layer, :] = centroid_vector_positive
        centroid_negative_true_pca_all[layer, :] = centroid_negative_true
        centroid_negative_false_pca_all[layer, :] = centroid_negative_false
        centroid_negative_vector_pca_all[layer, :] = centroid_vector_negative
    # # plot
    if save_plot:

        line_width = 3
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Original High Dimensional Space', 'PCA'))
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=cos_positive_negative,
                                 mode='lines+markers',
                                 showlegend=False,
                                 line=dict(color="royalblue", width=line_width),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
                                 x=np.arange(n_layers), y=cos_positive_negative_pca,
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
        fig['layout']['yaxis2']['range'] = [-1, 1.2]

        fig.show()
        # fig.write_html(save_path + os.sep + 'cos_sim_positive_negative.html')
        pio.write_image(fig, save_path + os.sep + 'stage_3_cos_sim_positive_negative_' +
                        f'_{contrastive_label[0]}_{contrastive_label[1]}.png',
                        scale=6)
    stage_3 = {
               'centroid_positive_true_pca_all': centroid_positive_true_pca_all,
               'centroid_positive_false_pca_all': centroid_positive_false_pca_all,
               'centroid_positive_vector_pca_all': centroid_positive_vector_pca_all,
               'centroid_negative_true_pca_all': centroid_negative_true_pca_all,
               'centroid_negative_false_pca_all': centroid_negative_false_pca_all,
               'centroid_negative_vector_pca_all': centroid_negative_vector_pca_all,
               'cos_positive_negative': cos_positive_negative,
               'cos_positive_negative_pca': cos_positive_negative_pca
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
                             contrastive_label=['HHH', 'evil_confidant'], save_plot=True):
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

    # 1. Stage 1: Separation between positive and negative
    # Measurement: The distance between a pair of positive and negative prompt
    # Future: Measure the within group (negative and positive) vs across group distance
    stage_1 = get_distance_pair_contrastive(activations_all, activations_pca, n_layers, save_path,
                                             contrastive_label=contrastive_label, save_plot=save_plot)

    # 2. Stage 2:  Separation between True and False
    # Measurement: the distance between centroid between centroids of true and false
    activations_all_dunn = torch.stack((activations_positive,
                                        activations_negative), dim=0)
    activations_all_dunn = torch.stack((activations_positive,
                                        activations_negative), dim=0)
    activations_pca_dunn = get_pca_layer_by_layer_dunn(activations_all_dunn,
                                                       n_components=3)
    labels_dunn = np.stack((labels, labels))
    stage_2 = get_dist_centroid_true_false(activations_all, activations_pca, labels, n_layers, save_path,
                                           contrastive_label=contrastive_label, save_plot=save_plot)

    stage_2_dunn = stage_2_get_distance_contrastive_dunn_index(activations_all_dunn, activations_pca_dunn,
                                                               labels_dunn,
                                                               contrastive_label, save_path)

    # 3. Stage 3: cosine similarity between the positive vector and negative vector
    # Measurement: cosine similarity between positive vector and negative vector 
    # positive vector is the centroid between positive true and positive false
    stage_3 = get_cos_sim_positive_negative_vector(activations_all, activations_pca, labels, n_layers, save_path,
                                              contrastive_label=contrastive_label, save_plot=save_plot)

    stage_stats = {
      'stage_1': stage_1,
      'stage_2': stage_2,
      'stage_2_dunn': stage_2_dunn,
      'stage_3': stage_3
    }

    return stage_stats


###################
def plot_stage_3_stats_original_intervention(cfg, stage_3_original, stage_3_intervention, n_layers,
                                             save_path, jailbreak_type):

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
                             x=np.arange(n_layers), y=stage_3_original['cos_positive_negative'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_intervention['cos_positive_negative'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width, dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_original['cos_positive_negative_pca'],
                             mode='lines+markers',
                             name="Original",
                             line=dict(color="royalblue", width=line_width)

    ), row=2, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_intervention['cos_positive_negative_pca'],
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
    pio.write_image(fig, save_path + os.sep + 'stage_3_cosine_similarity_layer_' + str(source_layer) + '_' +
                    str(target_layer_s) + '_' + str(target_layer_e) +
                    '_' + jailbreak_type + '.png',
                    scale=6)
    

def plot_stage_2_stats_original_intervention(cfg, stage_2_original, stage_2_intervention, n_layers,
                                             save_path, jailbreak_type):
    # plot stage 2
    line_width =3
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Original high dimensional space', 'PCA',
                                        '', '')
                        )

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_original['centroid_dist_positive'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_original['centroid_dist_negative'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="firebrick", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_intervention['centroid_dist_positive'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width, dash='dot')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_intervention['centroid_dist_negative'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="firebrick", width=line_width, dash='dot')
    ), row=1, col=1)
 
    # PCA
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_original['centroid_dist_positive_pca'],
                             mode='lines+markers',
                             name="Original_positive",
                             line=dict(color="royalblue", width=line_width)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_original['centroid_dist_negative_pca'],
                             mode='lines+markers',
                             name="Original_negative",
                             line=dict(color="firebrick", width=line_width)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_intervention['centroid_dist_positive_pca'],
                             mode='lines+markers',
                             name="Intervention_positive",
                             line=dict(color="royalblue", width=line_width, dash='dot')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_intervention['centroid_dist_negative_pca'],
                             mode='lines+markers',
                             name="Intervention_negative",
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
                    str(source_layer) + '_' + str(target_layer_s) + '_' + str(target_layer_e) +
                    '_' + jailbreak_type + '.png',
                    scale=6)
    
    
def plot_stage_1_stats_original_intervention(cfg, stage_1_original, stage_1_intervention, n_layers,
                                             save_path, jailbreak_type):
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
                    str(source_layer) + '_' + str(target_layer_s) + '_' + str(target_layer_e)
                    + '_' + jailbreak_type + '.png',
                    scale=6)


def plot_stage_quantification_original_intervention(cfg, stage_stats_original, stage_stats_intervention,
                                                    n_layers, save_path, jailbreak_type):

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

    plot_stage_1_stats_original_intervention(cfg, stage_1_original, stage_1_intervention, n_layers,
                                             save_path, jailbreak_type)
    plot_stage_2_stats_original_intervention(cfg, stage_2_original, stage_2_intervention, n_layers,
                                             save_path, jailbreak_type)
    plot_stage_3_stats_original_intervention(cfg, stage_3_original, stage_3_intervention, n_layers,
                                             save_path, jailbreak_type)

